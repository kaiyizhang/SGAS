import h5py
import numpy as np
import os
import random
import torch
import torch.utils.data as data
from tqdm import tqdm
from visdom import Visdom


import sys 
sys.path.append("..") 
from utils.utils import sample_point_cloud_by_n, plot_diff_pcds


part_id2name = {
    'Chair': {
        0: 'chair_back',
        1: 'chair_arm',
        2: 'chair_seat',
        3: 'chair_base'
    },
    'Lamp': {
        0: 'lamp_base',
        1: 'lamp_body',
        2: 'lamp_unit'
    },
    'Table': {
        0: 'tabletop',
        1: 'table_base'
    },
    'Bag': {
        0: 'bag_body',
        1: 'handle'
    },
    'Display': {
        0: 'display_screen',
        1: 'base'
    },
    'Earphone': {
        0: 'head_band',
        1: 'earcup_unit'
    },
    'Faucet': {
        0: 'frame',
        1: 'spout',
        2: 'switch',
    },
    'Hat': {
        0: 'crown',
        1: 'brim'
    },
    'Knife': {
        0: 'blade_side',
        1: 'handle_side'
    },
    'Laptop': {
        0: 'screen_side',
        1: 'base_side'
    },
    'Mug': {
        0: 'body',
        1: 'handle',
        2: 'supporting_plate',
        3: 'containing_things'
    }
}

def get_dataloader_partnet(phase, config):
    is_shuffle = phase == 'train'
    batch_size = config.batch_size if phase == 'train' else 1

    dataset = Partdataset(config.dataroot, config.class_choice, phase, config.part_id)
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=is_shuffle,
                                                num_workers=config.nThreads,
                                                worker_init_fn=np.random.seed())
    return dataloader


class Partdataset(data.Dataset):
    def __init__(self, dataroot=None, class_choice=None, phase='train', part_id=None):
        super(Partdataset, self).__init__()
        self.path = os.path.join(dataroot, class_choice, phase)
        self.item = [] # [(id, pcd, mask), ...]
        self.wo_semantic_pcd = [] # [(pcd), ...]
        self.phase = phase
        self.part_num = len(part_id2name[class_choice])
        self.part_pnum = 2048//self.part_num
        part_name2id = {v:k for k, v in part_id2name[class_choice].items()}
        self.part_id = part_id

        if part_id is not None:
            part_name = part_id2name[class_choice][part_id]
            for root, dirs, files in os.walk(self.path):
                print('initing %s dataset...' % (phase))
                for name in tqdm(files):
                    # print(os.path.join(root, name))
                    with h5py.File(os.path.join(root, name), 'r') as f:
                        if part_name not in f.keys():
                            # print(name.split('.')[0])
                            continue
                        else:
                            pcd = sample_point_cloud_by_n(f[part_name][()], self.part_pnum)
                            self.item.append((name.split('.')[0], pcd, np.ones(1)))
        else:
            for root, dirs, files in os.walk(self.path):
                print('initing %s dataset...' % (phase))
                for name in tqdm(files):
                    pcd = np.zeros((self.part_num*self.part_pnum, 3))
                    mask = np.zeros(self.part_num)
                    # print(os.path.join(root, name))
                    with h5py.File(os.path.join(root, name), 'r') as f:
                        for key in f.keys():
                            if (key == class_choice.lower()):
                                self.wo_semantic_pcd.append((sample_point_cloud_by_n(f[key][()], 2048)))
                                continue
                            id = int(part_name2id[key])
                            pcd[id*self.part_pnum:(id+1)*self.part_pnum] = \
                                sample_point_cloud_by_n(f[key][()], self.part_pnum)
                            mask[id] = 1
                        self.item.append((name.split('.')[0], pcd, mask))
        self.rng = random.Random(1234)

    def random_rm_parts(self, item):
        id, pcd, mask = item
        part_ids = np.argwhere(mask == 1).reshape(-1)
        if self.phase == "train":
            random.shuffle(part_ids)
            n_part_keep = random.randint(1, max(1, len(part_ids) - 1))
        else:
            self.rng.shuffle(part_ids)
            n_part_keep = self.rng.randint(1, max(1, len(part_ids) - 1))
        part_ids_keep = part_ids[:n_part_keep]

        raw_id = id
        raw_pcd = np.zeros((self.part_num*self.part_pnum, 3))
        raw_mask = np.zeros(self.part_num)
        partial_pcd = np.zeros((0, 3))
        for i in part_ids_keep:
            raw_pcd[self.part_pnum*i:self.part_pnum*(i+1)] = pcd[self.part_pnum*i:self.part_pnum*(i+1)]
            raw_mask[i] = 1
            partial_pcd = np.concatenate((partial_pcd, pcd[self.part_pnum*i:self.part_pnum*(i+1)]), axis=0)
        partial_pcd = sample_point_cloud_by_n(partial_pcd, 1024)
        return raw_id, raw_pcd, raw_mask, partial_pcd, pcd, n_part_keep
    
    def __getitem__(self, index):
        if self.part_id is not None:
            id, pcd, _ = self.item[index]
            # id, pcd, _ = self.item[random.randint(0, len(self.item) - 1)] # for test visual
            return {'id': id, 'pcd': torch.from_numpy(pcd).float()}
        else:
            raw_id, raw_pcd, raw_mask, partial_pcd, complete_pcd, n_part_keep = self.random_rm_parts(self.item[index])
            gt_pcd = self.wo_semantic_pcd[index]
            real_id, real_pcd, real_mask = self.item[random.randint(0, len(self.item) - 1)]
            # real_id, real_pcd, real_mask = self.item[index]
            return {'raw_id': raw_id, 
                    'raw_pcd': torch.from_numpy(raw_pcd).float(), 
                    'raw_mask': torch.from_numpy(raw_mask).float(),
                    'gt_pcd': torch.from_numpy(gt_pcd).float(),
                    'partial_pcd': torch.from_numpy(partial_pcd).float(),
                    'complete_pcd': torch.from_numpy(complete_pcd).float(),
                    'real_id': real_id, 
                    'real_pcd': torch.from_numpy(real_pcd).float(), 
                    'real_mask': torch.from_numpy(real_mask).float(),
                    'n_part_keep': n_part_keep}

    def __len__(self):
        return len(self.item)


if __name__ == "__main__":
    d = Partdataset(dataroot='../data/PartNet.v0.Merged', class_choice='Chair', phase='train')
    print(d)
    print(d[0])
    print(d[0]['gt_pcd'].shape)
    
    vis = Visdom(env='dataset')

    for i in range(20):
        plot_diff_pcds([d[i]['gt_pcd']],
            vis=vis,
            title=d[i]['raw_id'],
            legend=['pcd'],
            win='dataset_'+str(i))