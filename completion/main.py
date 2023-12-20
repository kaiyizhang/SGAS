import random
import os
from tqdm import tqdm
import numpy as np

import sys
sys.path.append("..")

from agent.part_ae import PartAEAgent
from agent.part_comp import PartCompAgent
from dataset.procpartnet import get_dataloader_partnet
from utils.utils import *


def train():
    dataloader = get_dataloader_partnet('train', config)
    val_dataloader = get_dataloader_partnet('val', config)

    if config.module == 'part_ae':
        agent = PartAEAgent(config)
    if config.module == 'part_comp':
        agent = PartCompAgent(config)

    for epoch in tqdm(range(1, config.n_epochs+1)):
        for i, data in enumerate(dataloader):
            agent.train_model(data=data, args=[epoch, i+1])
        if config.module == 'part_ae':
            for i, data in enumerate(val_dataloader):
                agent.val_model(data=data)
        agent.after_one_epoch(args=[epoch])

def test():
    random.seed(1856)
    dataloader = get_dataloader_partnet('test', config)

    agent = PartCompAgent(config)
    for i, data in enumerate(tqdm(dataloader)):
        if not config.vis:
            agent.test_model(data=data)
        else:
            agent.visual_model(data=data)

def metric():
    res_path = os.path.join('/'.join(config.gan_ckpt_path.split('/')[:-2]), 'results')
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    random.seed(1856)
    ref_dataloader = get_dataloader_partnet('train', config)
    dataloader = get_dataloader_partnet('test', config)

    print('getting ref pcds...')
    ref_pcds = [] # faster than torch.cat()
    for i, data in enumerate(tqdm(ref_dataloader)):
        ref_pcds.append(data['gt_pcd'])
    ref_pcds = torch.cat(ref_pcds, dim=0)
    # print(ref_pcds.shape)

    agent = PartCompAgent(config)

    # mmd_threshold
    save_file = os.path.join(res_path, 'mmd_threshold.txt')
    if not os.path.exists(save_file):
        for i, data in enumerate(tqdm(dataloader)):
            agent.get_mmd_threshold(data=data, ref=ref_pcds)

    # tmd_mmd
    min_mmd_list, min_thmmd, max_thmmd = agent.get_thmmd()
    # th = np.linspace(min_thmmd, max_thmmd, 10).tolist()
    th = np.linspace(0.5, 5, 10).tolist()
    tmd_list = []
    for i in range(len(th)):
        tmd = 0
        for j, data in enumerate(tqdm(dataloader)):
            if th[i] <= min_mmd_list[j]:
                # print(data['raw_id'][0])
                continue
            else:
                tmd += agent.get_tmd(data=data, ref=ref_pcds, th=th[i])
        tmd_list.append(tmd/len(min_mmd_list))
        print(tmd_list)

    with open(os.path.join(res_path, 'tmd_mmd.txt'), 'w') as f:
        for i in range(len(th)):
            f.write('%.3f' % (th[i]))
            f.write('\n') if i == len(th)-1 else f.write(' ')
        for i in range(len(tmd_list)):
            f.write('%.3f' % (tmd_list[i]))
            f.write('\n') if i == len(tmd_list)-1 else f.write(' ')


def realscan():
    random.seed(1856)
    agent = PartCompAgent(config)
    sample_id = config.sample_id
    
    if sample_id == "1":
        mask = [0,0,1,0]
    if sample_id == "2":
        mask = [1,0,1,0]
    if sample_id == "3":
        mask = [0,0,0,1]
    if sample_id == "4":
        mask = [0,1]
    if sample_id == "5":
        mask = [1,0]
    if sample_id == "6":
        mask = [1,0]
    
    pcd = np.load("../data/ScanNet/sample_%s.npy" % (sample_id))
    raw_id = ['sample_%s' % (sample_id)]

    # prepare data
    pcd = torch.from_numpy(sample_point_cloud_by_n(pcd, 2048)).float()
    raw_mask = mask
    raw_pcd = pcd
    raw_mask = torch.tensor(raw_mask).float()
    partial_pcd = torch.from_numpy(sample_point_cloud_by_n(pcd.detach().cpu().numpy(), 1024)).float()
    agent.save_realscan({'raw_pcd': raw_pcd.unsqueeze(0),
                        'raw_mask': raw_mask.unsqueeze(0), 
                        'raw_id': raw_id,
                        'partial_pcd': partial_pcd.unsqueeze(0)})

config = Config()
if config.task == 'train':
    train()
if config.task == 'test':
    test()
if config.task == 'metric':
    metric()
if config.task == "realscan":
    realscan()
