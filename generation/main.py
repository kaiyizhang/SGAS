from tqdm import tqdm
import os
import time

import sys
sys.path.append("..")

from agent.part_ae import PartAEAgent
from agent.part_gen import PartGenAgent
from agent.struct_gen import StructGenAgent
from dataset.shapenetseg import *
from utils.utils import *
from visdom import Visdom


def train():
    if config.module == 'struct_gen':
        loaders = get_data_loaders(config)
        train_loader, val_loader = loaders['train_loader'], loaders['test_loader']

        agent = StructGenAgent(config)
        for epoch in tqdm(range(1, config.n_epochs+1)):
            for i, data in enumerate(train_loader):
                agent.train_model(data=data, args=[epoch, i+1])
            for i, data in enumerate(val_loader):
                agent.val_model(data=data)
            agent.after_one_epoch(args=[epoch])
    else:
        _, dataloader = get_data_loaders_proc('train', config)
        _, val_dataloader = get_data_loaders_proc('val', config)

        if config.module == 'part_ae':
            agent = PartAEAgent(config)
            for epoch in tqdm(range(1, config.n_epochs+1)):
                for i, data in enumerate(dataloader):
                    agent.train_model(data=data, args=[epoch, i+1])
                if config.module == 'part_ae':
                    for i, data in enumerate(val_dataloader):
                        agent.val_model(data=data)
                agent.after_one_epoch(args=[epoch])
        
        if config.module == 'part_gen':
            agent = PartGenAgent(config)

            # get all ref pcds
            ref_pcs = []
            for i, data in enumerate(tqdm(val_dataloader)):
                ref_pcs.append(data['gt_pcd'])
            ref_pcs = torch.cat(ref_pcs, dim=0).to(config.device)

            for epoch in tqdm(range(1, config.n_epochs+1)):
                for i, data in enumerate(dataloader):
                    agent.train_model(data=data, args=[epoch, i+1])
                agent.after_one_epoch(args=[epoch], ref_pcs=ref_pcs)

def test():
    _, test_dataloader = get_data_loaders_proc('test', config)
    agent = PartGenAgent(config)

    if not config.vis:
        # get all ref pcds
        ref_pcs = []
        for i, data in enumerate(tqdm(test_dataloader)):
            ref_pcs.append(data['gt_pcd'])
        ref_pcs = torch.cat(ref_pcs, dim=0).to(config.device)

        # calc metrics
        rs = 100  # rs rounds to get mean
        for r in range(rs):
            agent.calc_metrics(ref_pcs=ref_pcs)
        print('mean:')
        for k, v in agent.metrics.items():
            print(k+':', v/rs)
    else:
        for i, data in enumerate(tqdm(test_dataloader)):
            agent.visual_model(data=data)
            if i == 10:
                exit()
    
def upartseg():
    if not config.vis:
        dataset = get_datasets(config)
        agent = StructGenAgent(config)

        print('processing %s dataset...' % (config.class_choice))
        for i in tqdm(range(len(dataset))):
            _, parts = agent.inference(data=dataset[i])
            parts = parts.squeeze()

            # save_parts
            save_dir = os.path.join(config.dataroot, cate_to_synsetid[config.class_choice], 
                                    'proc', 'k'+str(config.part_num)+'_ext'+str(config.ext_ratio))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with h5py.File(os.path.join(save_dir, dataset[i]['token']+'.h5'), 'w') as f:
                f.create_dataset('raw', data=dataset[i]['point_set'].detach().cpu().numpy())
                for p in range(parts.shape[0]):
                    f.create_dataset('p'+str(p), data=parts[p].detach().cpu().numpy())
    else:
        vis = Visdom(env=config.visdom_env)

        read_dir = os.path.join(config.dataroot, cate_to_synsetid[config.class_choice], 
                                'proc', 'k'+str(config.part_num)+'_ext'+str(config.ext_ratio))
        for root, _, files in os.walk(read_dir):
            for name in files:
                print(name)
                res = []
                with h5py.File(os.path.join(read_dir, name), 'r') as f:
                    for ke in f.keys():
                        res.append(torch.from_numpy(f[ke][()]).float())
                
                plot_diff_pcds(res,
                                vis=vis,
                                title=name,
                                legend=[str(i) for i in range(len(res)-1)]+['raw'],
                                win='visual')
                time.sleep(5)


config = Config()
if config.task == 'train':
    train()
if config.task == 'test':
    test()
if config.task == 'upartseg':
    upartseg()
      