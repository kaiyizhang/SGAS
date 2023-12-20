from datetime import datetime, timedelta
import numpy as np
import os
import sys
import torch

import sys 
sys.path.append("..")
from utils.utils import *

sys.path.append('../utils/PyTorchEMD/')
from emd import earth_mover_distance
sys.path.append('../utils/pyTorchChamferDistance/chamfer_distance/')
from chamfer_distance import chamfer_distance


class BaseAgent:
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """
    def __init__(self, config):
        self.config = config
    
    def loss_emd(self, array1, array2):
        dist = earth_mover_distance(array1, array2, transpose=False)
        # print(dist.shape)
        dist = torch.mean(dist) / array1.shape[1]
        return dist
    
    def eval_emd(self, array1, array2):
        dist = earth_mover_distance(array1, array2, transpose=False)
        dist = torch.mean(dist) / array1.shape[1]
        return dist * 100

    def loss_cd(self, array1, array2):
        dist1, dist2 = chamfer_distance(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist

    def eval_cd(self, array1, array2):
        dist1, dist2 = chamfer_distance(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist * 10000

    def init_log(self):
        now = (datetime.utcnow()+timedelta(hours=8)).isoformat()
        self.config.save_path = \
            os.path.join('./log', self.config.class_choice, 'k'+str(self.config.part_num), self.config.module, now)
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)
        os.system('cp config.json %s' % (self.config.save_path))
        os.system('cp agent/%s.py %s' % (self.config.module, self.config.save_path))

    def loss_curves(self, losses):
        if not hasattr(self, 'curve_data'):
            self.curve_data = {'X':[], 'Y':[], 'legend':list(losses.keys())}
        self.curve_data['X'].append(self.epoch)
        self.curve_data['Y'].append([losses[k] for k in self.curve_data['legend']])
            
        self.vis.line(
            X=np.array(self.curve_data['X']),
            Y=np.array(self.curve_data['Y']),
            opts={
                'title': 'runing loss over time',
                'legend': self.curve_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win='loss_curves')

    def extend_parts(self, parts, part_pnum, device):
        # extend parts to fixed number of points
        ext_parts = []
        for i in range(len(parts)):
            one = parts[i]
            others = torch.cat(parts[:i]+parts[i+1:], dim=0)

            if one.shape[0] >= part_pnum:
                one = torch.from_numpy(sample_point_cloud_by_n(one.detach().cpu().numpy(), \
                    part_pnum)).unsqueeze(0).to(device)  # 1*part_pnum*3
                ext_parts.append(one)
            else:
                ext_pnum = part_pnum - one.shape[0]
                _one = one.unsqueeze(0).repeat(others.shape[0], 1, 1)
                _others = others.unsqueeze(1).repeat(1, one.shape[0], 1)
                pair_dis = torch.sum((_one - _others)**2, dim=2)
                _, idx = torch.topk(torch.min(pair_dis, dim=1)[0], \
                    k=ext_pnum, largest=False)  # others.shape[0]
                ext_p = torch.index_select(others, 0, idx)
                one = torch.cat((one, ext_p), dim=0).unsqueeze(0)
                ext_parts.append(one)
        ext_parts = torch.cat(ext_parts, dim=0).unsqueeze(0)  # 1*N*part_pnum*3
        return ext_parts

    def partcluster(self, structs: torch.tensor, pcds: torch.tensor):
        """
        :param: structs: (B, N, 3)
        :param: pcds: (B, 2048, 3)
        :return: (B, N, 2048//N*(1+ext_ratio), 3)
        """
        B, N = structs.shape[0], structs.shape[1]
        part_pnum = int(2048//N*(1+self.config.ext_ratio))

        _structs = structs.unsqueeze(1).repeat(1, 2048, 1, 1)
        _pcds = pcds.unsqueeze(2).repeat(1, 1, N, 1)

        pair_dis = torch.sum((_structs - _pcds)**2, dim=3)  # B*2048*N
        idx = torch.argmin(pair_dis, dim=2)  # B*2048

        origin_parts = []
        batch_parts = []
        for b in range(B):
            # parts that divide up all the points
            parts = []
            for n in range(N):
                _idx = torch.nonzero(idx[b] == n, as_tuple=False).squeeze()
                part = torch.index_select(pcds[b], 0, _idx)
                # print(part.shape)
                parts.append(part)
            origin_parts.append(parts)

            ext_parts = self.extend_parts(parts, part_pnum, structs.device)
            batch_parts.append(ext_parts)

        batch_parts = torch.cat(batch_parts, dim=0)
        return origin_parts, batch_parts
