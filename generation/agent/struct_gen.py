import torch
import torch.nn as nn
import os
from visdom import Visdom

import sys
sys.path.append("..")

from agent.base import BaseAgent
from utils.utils import *
from networks import *


class StructGenAgent(BaseAgent):
    def __init__(self, config):
        super(StructGenAgent, self).__init__(config)
        self.config = config
        self.build_net()

        if self.config.task == 'train':
            self.vis = Visdom(env=config.visdom_env)
            self.init_log()
            self.set_optimizer()
            self.best_cd = 0
            self.cd = {config.class_choice: {'num': 0, 'value': 0}}
        else:
            self.vis = Visdom(env=config.visdom_env)
    
    def build_net(self):
        self.sGen = nn.DataParallel(
            StructGen(self.config).to(self.config.device), device_ids=self.config.gpu_ids)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.sGen.parameters()), lr=self.config.lr, betas=(0.9, 0.999))

    def train_model(self, data, args):
        self.token = data['token']
        self.input = data['point_set'].to(self.config.device)

        self.sGen.train()
        self.output = self.sGen(self.input)
        self.loss = self.loss_cd(self.output, self.input)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def val_model(self, data):
        self.input_val = data['point_set'].to(self.config.device)
        B = self.input_val.shape[0]
        # print(self.input_val.shape)
    
        with torch.no_grad():
            self.sGen.eval()
            self.output_val = self.sGen(self.input_val)

        value = self.eval_cd(self.output_val, self.input_val)
        self.cd[self.config.class_choice]['num'] += B
        self.cd[self.config.class_choice]['value'] += value * B

    def save_ckpt(self, step):
        save_path = os.path.join(self.config.save_path, 'ckpts')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.epoch % step == 0:
            torch.save({'epoch': self.epoch, 'state_dict': self.sGen.state_dict()}, os.path.join(save_path, 'epoch_%s.pth' % (self.epoch)))
        if self.epoch == 1 or self.cd[self.config.class_choice]['value'] <= self.best_cd:
            self.best_cd = self.cd[self.config.class_choice]['value']
            torch.save({'epoch': self.epoch, 'state_dict': self.sGen.state_dict()}, os.path.join(save_path, 'epoch_best.pth'))

    def after_one_epoch(self, args):
        self.epoch = args[0]
        class_choice = self.config.class_choice

        if self.config.vis:
            plot_diff_pcds([self.input[0], self.output[0]],
                                    vis=self.vis,
                                    title=self.token[0],
                                    legend=['input', 'output'],
                                    win='visual')

        self.cd[class_choice]['value'] /= self.cd[class_choice]['num']
        losses = {'loss': self.loss.item()}
        losses['val_'+class_choice] = self.cd[class_choice]['value'].item()
        self.loss_curves(losses)

        self.save_ckpt(step=self.config.save_frequency)
        self.cd = {class_choice: {'num': 0, 'value': 0}}

    def load_ckpts(self):
        ckpt = torch.load(self.config.axform_ckpt_path, map_location=lambda storage, location: storage)
        # print("load epoch: %d" % (ckpt['epoch']))
        self.sGen.load_state_dict(ckpt['state_dict'])

    def inference(self, data):
        self.token = data['token']
        self.pcds = data['point_set'].unsqueeze(0).to(self.config.device)  # 1*2048*3
        self.load_ckpts()

        with torch.no_grad():
            self.sGen.eval()
            self.structs = self.sGen(self.pcds)
        
        origin_parts, parts = self.partcluster(self.structs, self.pcds)
        return origin_parts, parts  # (1, N, 2048//N*(1+ext_ratio),3)
