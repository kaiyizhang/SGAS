import h5py
import os
import torch
import numpy as np
from torch.utils import data
from tqdm import tqdm
import random
from pprint import pprint
from tqdm import tqdm
from visdom import Visdom
import sys
sys.path.append("..")

from utils.utils import sample_point_cloud_by_n

synsetid_to_cate = {
    '02691156': 'Airplane', '02773838': 'Bag',
    '02954340': 'Cap', '02958343': 'Car', '03001627': 'Chair',
    '03261776': 'Earphone',
    '03467517': 'Guitar',
    '03624134': 'Knife', '03636649': 'Lamp', '03642806': 'Laptop',
    '03790512': 'Motorbike', '03797390': 'Mug',
    '03948459': 'Pistol',
    '04099429': 'Rocket', '04225987': 'Skateboard', '04379243': 'Table'
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class BenchmarkDataset(data.Dataset):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None, sample=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.uniform = uniform
        self.sample = sample
        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
                
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            # dir_sampling = os.path.join(self.root, self.cat[item], 'sampling')

            fns = sorted(os.listdir(dir_point))

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                # self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'), os.path.join(dir_sampling, token + '.sam')))
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # self.datapath.append((item, fn[0], fn[1], fn[2]))
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l

        print('initing dataset...')
        self.item = []
        for fn in tqdm(self.datapath):
            cls = self.classes[fn[0]]
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64)
            token = fn[1].split('/')[-1].split('.')[0]
            self.item.append((cls, point_set, seg, token))

    def __getitem__(self, index):
        # fn = self.datapath[index]
        # cls = self.classes[self.datapath[index][0]]
        # point_set = np.loadtxt(fn[1]).astype(np.float32)
        # seg = np.loadtxt(fn[2]).astype(np.int64)
        cls, point_set, seg, token = self.item[index]

        if self.uniform:
            # choice = np.loadtxt(fn[3]).astype(np.int64)
            if point_set.shape[0] >= 2048:
                choice = random.sample(range(point_set.shape[0]), 2048)
            else:
                r = 2048 // point_set.shape[0]
                choice = random.sample(range(point_set.shape[0]), 2048-r*point_set.shape[0])
                for i in range(r):
                    choice += list(range(point_set.shape[0]))
            assert len(choice) == self.npoints, "Need to match number of choice(2048) with number of vertices."
        else:
            # print(len(seg))
            choice = np.random.randint(0, len(seg), size=self.npoints)

        if self.sample:
            point_set = point_set[choice]
            seg = seg[choice]

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        # print(seg, seg.shape, torch.max(seg))

        if self.classification:
            return {'point_set': point_set, 'cls': cls, 'token': token}
        else:
            return {'point_set': point_set, 'seg': seg, 'token': token}

    def __len__(self):
        return len(self.datapath)

def get_datasets(args):
    dataset = BenchmarkDataset(
        root=args.dataroot, npoints=2048, uniform=True, class_choice=args.class_choice)
    print("Dataset : {} prepared.".format(len(dataset)))
    return dataset

def get_data_loaders(args):
    dataset = get_datasets(args)
    train_loader = data.DataLoader(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.nThreads)
    test_loader = data.DataLoader(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=False, pin_memory=True, num_workers=args.nThreads)

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders


#############################################################
# ShapeNet Dataset with segmented parts
#############################################################

def get_data_loaders_proc(phase, config):
    is_shuffle = phase == 'train'

    dataset = Procdataset(config, config.dataroot, config.class_choice, config.part_id)
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=config.batch_size,
                                                shuffle=is_shuffle,
                                                num_workers=config.nThreads,
                                                worker_init_fn=np.random.seed())
    return dataset, dataloader


class Procdataset(data.Dataset):
    def __init__(self, config, dataroot=None, class_choice=None, part_id=None):
        super(Procdataset, self).__init__()
        self.config = config
        self.catfile = os.path.join(dataroot, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]   
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.path = os.path.join(dataroot, self.cat[class_choice], 'proc', 
                                 'k'+str(config.part_num)+'_ext'+str(config.ext_ratio))
        self.item = [] # [(id, pcd), ...]
        self.wo_semantic_pcd = [] # [(pcd), ...]
        self.part_id = part_id
        self.part_pnum = int(2048//config.part_num*(1+config.ext_ratio))
        
        if part_id is not None:
            for root, dirs, files in os.walk(self.path):
                print('initing dataset...')
                for name in tqdm(files):
                    with h5py.File(os.path.join(root, name), 'r') as f:
                        pcd = f['p'+str(part_id)][()]
                        self.item.append((name.split('.')[0], pcd))
        else:
            for root, dirs, files in os.walk(self.path):
                print('initing dataset...')
                for name in tqdm(files):
                    parts = {}
                    with h5py.File(os.path.join(root, name), 'r') as f:
                        for key in f.keys():
                            if (key == 'raw'):
                                self.wo_semantic_pcd.append(f[key][()])
                                continue
                            id = int(key[1:])
                            parts[id] = f[key][()]
                    pcd = []
                    for i in range(len(parts)):
                        pcd.append(parts[i])
                    pcd = np.concatenate(pcd, axis=0)
                    self.item.append((name.split('.')[0], pcd))

    def random_rm_parts(self, item):
        id, pcd = item
        part_ids = np.arange(self.config.part_num)
        random.shuffle(part_ids)
        n_part_keep = random.randint(1, max(1, len(part_ids) - 1))
        part_ids_keep = part_ids[:n_part_keep]

        raw_id = id
        raw_pcd = np.zeros((self.config.part_num*self.part_pnum, 3))
        raw_mask = np.zeros(self.config.part_num)
        partial_pcd = np.zeros((0, 3))
        for i in part_ids_keep:
            raw_pcd[self.part_pnum*i:self.part_pnum*(i+1)] = pcd[self.part_pnum*i:self.part_pnum*(i+1)]
            raw_mask[i] = 1
            partial_pcd = np.concatenate((partial_pcd, pcd[self.part_pnum*i:self.part_pnum*(i+1)]), axis=0)
        partial_pcd = sample_point_cloud_by_n(partial_pcd, 1024)
        return raw_id, raw_pcd, raw_mask, partial_pcd, pcd, n_part_keep

    def __getitem__(self, index):
        if self.part_id is not None:
            id, pcd = self.item[index]
            return {'id': id, 'pcd': torch.from_numpy(pcd).float()}
        else:
            raw_id, raw_pcd, raw_mask, partial_pcd, complete_pcd, n_part_keep = self.random_rm_parts(self.item[index])
            gt_pcd = self.wo_semantic_pcd[index]
            real_id, real_pcd = self.item[random.randint(0, len(self.item) - 1)]
            return {'raw_id': raw_id, 
                    'raw_pcd': torch.from_numpy(raw_pcd).float(), 
                    'raw_mask': torch.from_numpy(raw_mask).float(),
                    'gt_pcd': torch.from_numpy(gt_pcd).float(),
                    'partial_pcd': torch.from_numpy(partial_pcd).float(),
                    'complete_pcd': torch.from_numpy(complete_pcd).float(),
                    'real_id': real_id, 
                    'real_pcd': torch.from_numpy(real_pcd).float(),
                    'n_part_keep': n_part_keep}

    def __len__(self):
        return len(self.item)


if __name__ == "__main__":
    import sys 
    sys.path.append("..")
    from utils.utils import plot_diff_pcds

    vis = Visdom(env='dataset')
    class_choice='Airplane'
    d1 = BenchmarkDataset(root='../data/shapenetcore_partanno_segmentation_benchmark_v0', 
                            npoints=2048, uniform=True, classification=False, class_choice=class_choice, sample=False)
    
    print(d1[0], d1[0]['point_set'].shape)
    pcd = []
    for i in range(10):
        pcd.append(d1[i]['point_set'])
    plot_diff_pcds(pcd, vis=vis, title='test', legend=[str(i) for i in range(10)], win='test')
    
    # max_seg = 0
    # for i in range(len(d1)):
    #     res = torch.max(d1[i]['seg'])
    #     if res > max_seg:
    #         max_seg = res
    # print(max_seg) # airplane 4; chair 4; table 3


    # process original shapenetseg dataset
    # for i in tqdm(range(len(d1))):
    #     pcd = d1[i]['point_set']
    #     seg = d1[i]['seg']
    #     token = d1[i]['token']

    #     parts = {}
    #     seg_num = torch.unique(seg).shape[0]
    #     for j in range(1, seg_num+1):
    #         idx = torch.nonzero(seg == j, as_tuple=False).squeeze()
    #         part = torch.index_select(pcd, 0, idx)
    #         # print(part.shape)
    #         if part.shape[0] == 0:
    #             continue
    #         else:
    #             parts['p'+str(j-1)] = part
    #     # pprint(parts)
            
    #     save_path = os.path.join('../data/shapenetcore_partanno_segmentation_benchmark_v0', 
    #                                 cate_to_synsetid[class_choice], 'sem_parts')
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     with h5py.File(os.path.join(save_path, token+'.h5'), 'w') as f:
    #         f.create_dataset('raw', data=pcd.detach().cpu().numpy())
    #         for k, v in parts.items():
    #             f.create_dataset(k, data=v.detach().cpu().numpy())
