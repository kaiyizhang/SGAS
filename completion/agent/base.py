from datetime import datetime, timedelta
import numpy as np
import os
import sys
import torch

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

    # Taken from https://github.com/stevenygd/PointFlow
    def _pairwise_CD_(self, sample_pcs, ref_pcs, batch_size):
        N_sample = sample_pcs.shape[0]
        N_ref = ref_pcs.shape[0]
        all_cd = []
        for sample_b_start in range(N_sample):
            sample_batch = sample_pcs[sample_b_start]

            cd_lst = []
            for ref_b_start in range(0, N_ref, batch_size):
                ref_b_end = min(N_ref, ref_b_start + batch_size)
                ref_batch = ref_pcs[ref_b_start:ref_b_end]

                batch_size_ref = ref_batch.size(0)
                sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
                sample_batch_exp = sample_batch_exp.contiguous()

                dl, dr = chamfer_distance(sample_batch_exp, ref_batch)
                cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
            
            all_cd.append(torch.cat(cd_lst, dim=1))

        all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
        # print(all_cd.shape)
        return all_cd

    def mmd_cd(self, array1, array2):
        all_cd = self._pairwise_CD_(array1, array2, array2.shape[0])

        mmd_list, _ = torch.min(all_cd, dim=1) # original mmd's is dim=0
        mmd = mmd_list.mean()
        return mmd_list * 1000, mmd * 1000

    def tmd(self, array:list):
        sum_dist = 0
        for j in range(len(array)):
            for k in range(j + 1, len(array), 1):
                pc1 = array[j]
                pc2 = array[k]
                chamfer_dist = self.loss_cd(pc1, pc2).item()
                sum_dist += chamfer_dist
        mean_dist = sum_dist * 2 / (len(array) - 1)
        return mean_dist * 100

    def init_log(self):
        now = (datetime.utcnow()+timedelta(hours=8)).isoformat()
        self.config.save_path = \
            os.path.join('./log', self.config.class_choice, self.config.module, now)
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
