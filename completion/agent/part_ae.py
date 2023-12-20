import os
import torch
import torch.nn as nn
from visdom import Visdom

import sys 
sys.path.append("..")

from agent.base import BaseAgent
from utils.utils import *
from networks import *
from dataset.procpartnet import part_id2name


class PartAEAgent(BaseAgent):
    def __init__(self, config):
        super(PartAEAgent, self).__init__(config)
        self.config = config
        config.part_num = len(part_id2name[config.class_choice])
        config.part_pnum = 2048//config.part_num
        self.build_net()
        
        if self.config.task == 'train':
            self.vis = Visdom(env=config.visdom_env)
            self.init_log()
            self.set_optimizer()
            self.best_emd = 0
            self.emd = {config.class_choice: {'num': 0, 'value': 0}}
        else:
            self.vis = Visdom(env=config.visdom_env)
    
    def build_net(self):
        self.ae = nn.DataParallel(
            PartAE(self.config).to(self.config.device), device_ids=self.config.gpu_ids)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.ae.parameters()), lr=self.config.lr, betas=(0.9, 0.999))
    
    def train_model(self, data, args):
        self.id = data['id']
        self.input = data['pcd'].to(self.config.device)
        
        self.ae.train()
        self.output = self.ae(self.input)
        self.loss = self.loss_emd(self.output, self.input)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def val_model(self, data):
        self.id_val = data['id']
        self.input_val = data['pcd'].to(self.config.device)
    
        with torch.no_grad():
            self.ae.eval()
            self.output_val = self.ae(self.input_val)

        value = self.eval_emd(self.output_val, self.input_val)
        self.emd[self.config.class_choice]['num'] += 1
        self.emd[self.config.class_choice]['value'] += value

    def save_ckpts(self, step):
        save_path = os.path.join(self.config.save_path, 'ckpts')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.epoch % step == 0:
            torch.save({'epoch': self.epoch, 'state_dict': self.ae.state_dict()}, os.path.join(save_path, 'epoch_%d.pth' % (self.epoch)))
        if self.epoch == 1 or self.emd[self.config.class_choice]['value'] <= self.best_emd:
            self.best_emd = self.emd[self.config.class_choice]['value']
            torch.save({'epoch': self.epoch, 'state_dict': self.ae.state_dict()}, os.path.join(save_path, 'epoch_best.pth'))

    def after_one_epoch(self, args):
        self.epoch = args[0]
        class_choice = self.config.class_choice

        if self.config.vis:
            plot_diff_pcds([self.input[0], self.output[0]],
                                    vis=self.vis,
                                    title=self.id[0],
                                    legend=['input', 'output'],
                                    win='visual')

        self.emd[class_choice]['value'] /= self.emd[class_choice]['num']
        losses = {'loss': self.loss.item()}
        losses['val_'+class_choice] = self.emd[class_choice]['value'].item()
        self.loss_curves(losses)

        self.save_ckpts(step=self.config.save_frequency)
        self.emd = {class_choice: {'num': 0, 'value': 0}}

    def load_pretrained_AE(self):
        self.part_enc = nn.DataParallel(
            Pcd2Feat(self.config).to(self.config.device), device_ids=self.config.gpu_ids)
        self.part_dec = nn.DataParallel(
            Feat2Pcd(self.config).to(self.config.device), device_ids=self.config.gpu_ids)

        ckpts_path = self.config.ae_ckpt_path
        
        dict_new = self.part_enc.state_dict().copy()
        new_list = list(dict_new.keys())
        # print(new_list)
        n_paras = len(new_list) // self.config.part_num  # enc paras in one part
        for i in range(self.config.part_num):
            dict_trained = torch.load(ckpts_path[i], 
                            map_location=lambda storage, location: storage)['state_dict']
            trained_list = list(dict_trained.keys())
            # print(trained_list)
            for j in range(n_paras):
                dict_new[ new_list[j+i*n_paras] ] = \
                dict_trained[ trained_list[j+i*(len(trained_list)//self.config.part_num)] ]
        self.part_enc.load_state_dict(dict_new)

        
        dict_new = self.part_dec.state_dict().copy()
        new_list = list(dict_new.keys())
        # print(new_list)
        n_paras = len(new_list) // self.config.part_num
        for i in range(self.config.part_num):
            dict_trained = torch.load(ckpts_path[i], 
                            map_location=lambda storage, location: storage)['state_dict']
            trained_list = list(dict_trained.keys())
            # print(trained_list)
            for j in range(n_paras):
                dict_new[ new_list[j+i*n_paras] ] = \
                dict_trained[ trained_list[j+(i+1)*(len(trained_list)//self.config.part_num)-n_paras] ]
        self.part_dec.load_state_dict(dict_new)
        self.part_enc.eval()  # important!
        self.part_dec.eval()
