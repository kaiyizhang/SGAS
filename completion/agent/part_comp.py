import os
import torch
import torch.nn as nn
import time
import h5py
import shutil
import numpy as np
from visdom import Visdom

import sys 
sys.path.append("..")

from agent.part_ae import PartAEAgent
from utils.utils import *
from networks import *
from dataset.procpartnet import part_id2name


class PartCompAgent(PartAEAgent):
    def __init__(self, config):
        self.config = config
        config.part_num = len(part_id2name[config.class_choice])
        config.part_pnum = 2048//config.part_num
        self.load_pretrained_AE()
        self.build_net()

        if self.config.task == 'train':
            self.vis = Visdom(env=config.visdom_env)
            self.init_log()
            self.set_optimizer()
            self.loss_gp = GradientPenalty(lambdaGP=10, gamma=1, device=config.device)
        else:
            self.vis = Visdom(env=config.visdom_env)

    def build_net(self):
        self.netG = nn.DataParallel(
            GeneratorComp(self.config).to(self.config.device), device_ids=self.config.gpu_ids)
        self.netD = nn.DataParallel(
            Discriminator(self.config).to(self.config.device), device_ids=self.config.gpu_ids)
        self.netD_full = nn.DataParallel(
            DiscriminatorFull(self.config).to(self.config.device), device_ids=self.config.gpu_ids)

    def set_optimizer(self):
        self.optimizerG = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.netG.parameters()), lr=self.config.lr, betas=(0.5, 0.999))
        self.optimizerD = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.netD.parameters()), lr=self.config.lr, betas=(0.5, 0.999))
        self.optimizerD_full = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.netD_full.parameters()), lr=self.config.lr, betas=(0.5, 0.999))

    def generate_noise(self, B):
        self.z = torch.randn((B, self.config.noise_dim)).float().to(self.config.device)

    def get_real_latent(self):        
        self.real_latent = torch.empty(size=(self.real_pcd.shape[0], 0)).to(self.real_pcd.device)
        for i in range(self.config.part_num):
            self.real_latent = torch.cat((self.real_latent, \
                self.part_enc.module.parts[i](self.real_pcd[:, self.config.part_pnum*i:self.config.part_pnum*(i+1)])), dim=1)
        
        mask = self.real_mask.unsqueeze(2).repeat(1, 1, self.config.latent_dim).reshape(-1, self.config.latent_dim*self.config.part_num)
        self.real_latent = self.real_latent * mask

    def get_raw_latent(self):
        self.raw_latent = torch.empty(size=(self.raw_pcd.shape[0], 0)).to(self.raw_pcd.device)
        for i in range(self.config.part_num):
            self.raw_latent = torch.cat((self.raw_latent, \
                self.part_enc.module.parts[i](self.raw_pcd[:, self.config.part_pnum*i:self.config.part_pnum*(i+1)])), dim=1)
        
        mask = self.raw_mask.unsqueeze(2).repeat(1, 1, self.config.latent_dim).reshape(-1, self.config.latent_dim*self.config.part_num)
        self.raw_latent = self.raw_latent * mask
    
    def get_inputD_latent(self):
        mask = self.raw_mask.unsqueeze(2).repeat(1, 1, self.config.latent_dim).reshape(-1, self.config.latent_dim*self.config.part_num)
        self.inputD_latent = self.fake_latent * (1-mask) + self.raw_latent

    def train_model(self, data, args):
        self.real_pcd = data['real_pcd'].to(self.config.device)
        self.real_mask = data['real_mask'].to(self.config.device)
        self.real_id = data['real_id']
        self.raw_pcd = data['raw_pcd'].to(self.config.device)
        self.raw_mask = data['raw_mask'].to(self.config.device)
        self.raw_id = data['raw_id']
        self.partial_pcd = data['partial_pcd'].to(self.config.device)

        self.get_real_latent()  # 1*512
        self.get_raw_latent()  # 1*512
        self.generate_noise(self.real_pcd.shape[0])

        # update D network
        self.fake_latent = self.netG(self.z, self.partial_pcd)
        self.get_inputD_latent()

        real_out = self.netD(self.real_latent)
        fake_out = self.netD(self.fake_latent.detach())
        self.loss_D_real = -torch.mean(real_out)
        self.loss_D_fake = torch.mean(fake_out)
        self.loss_D_gp = self.loss_gp(self.netD, self.real_latent, self.fake_latent.detach())
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gp

        self.optimizerD.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizerD.step()

        real_out_full = self.netD_full(self.real_latent)
        fake_out_full = self.netD_full(self.inputD_latent.detach())
        self.loss_D_real_full = -torch.mean(real_out_full)
        self.loss_D_fake_full = torch.mean(fake_out_full)
        self.loss_D_gp_full = self.loss_gp(self.netD_full, self.real_latent, self.inputD_latent.detach())
        self.loss_D_full = self.loss_D_real_full + self.loss_D_fake_full + self.loss_D_gp_full
        
        self.optimizerD_full.zero_grad()
        self.loss_D_full.backward()
        self.optimizerD_full.step()

        if args[1] % 5 - 1 == 0:
            # update G network
            self.fake_latent = self.netG(self.z, self.partial_pcd)
            self.get_inputD_latent()

            fake_out = self.netD(self.fake_latent)
            wpartial_out = self.netD_full(self.inputD_latent)
            self.loss_G = self.config.g_ratio[0] * -torch.mean(fake_out)
            self.loss_G_full = self.config.g_ratio[1] * -torch.mean(wpartial_out)

            self.optimizerG.zero_grad()
            (self.loss_G+self.loss_G_full).backward()
            self.optimizerG.step()

    def save_ckpts(self, step):
        if self.epoch % step == 0:
            save_fn = 'epoch_%d.pth' % (self.epoch)
            save_dir = os.path.join(self.config.save_path, 'ckpts')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save({'epoch': self.epoch, 'state_dict': self.netG.state_dict()}, os.path.join(save_dir, save_fn))

    def after_one_epoch(self, args):
        self.epoch = args[0]

        fake_pcd_list = []
        for i in range(self.config.num_z):
            z = torch.randn((1, self.config.noise_dim)).float().to(self.config.device)
            with torch.no_grad():
                self.netG.eval()
                fake_latent = self.netG(z, self.partial_pcd[:1]) # 1*512

            fake_pcd = torch.empty(size=(fake_latent.shape[0], 0, 3)).to(fake_latent.device)
            for i in range(self.config.part_num):
                fake_pcd = torch.cat((fake_pcd, self.part_dec.module.parts[i](\
                    fake_latent[:, self.config.latent_dim*i:self.config.latent_dim*(i+1)])), dim=1)

            fake_pcd = self.raw_pcd[:1] + fake_pcd * (1 - self.raw_mask[:1].unsqueeze(2).repeat(1, 1, self.config.part_pnum) \
                .reshape(-1, self.config.part_num*self.config.part_pnum, 1).repeat(1, 1, 3))
            fake_pcd_list.append(fake_pcd[0])

        if self.config.vis:
            plot_diff_pcds(fake_pcd_list,
                                vis=self.vis,
                                title='epoch %d' % self.epoch,
                                legend=['fake'+str(i) for i in range(self.config.num_z)],
                                win='visual')

        losses = {'loss_G': self.loss_G.item(),
                  'loss_G_full': self.loss_G_full.item(),
                  'loss_D': self.loss_D.item(),
                  'loss_D_full': self.loss_D_full.item()}
        self.loss_curves(losses)

        self.save_ckpts(step=self.config.save_frequency)

    def load_ckpts(self):
        ckpt = torch.load(self.config.gan_ckpt_path, map_location=lambda storage, location: storage)
        # print("load epoch: %d" % (ckpt['epoch']))
        self.netG.load_state_dict(ckpt['state_dict'])

    def get_input_gen(self):
        self.gen_pcd = torch.empty(size=(self.fake_latent.shape[0], 0, 3)).to(self.fake_latent.device)
        self.input_pcd = torch.empty(size=(self.fake_latent.shape[0], 0, 3)).to(self.fake_latent.device)
        for j in range(self.config.part_num):
            if self.raw_mask[:, j] == 1:
                # continue
                self.input_pcd = torch.cat((self.input_pcd, self.raw_pcd[:, j*self.config.part_pnum:(j+1)*self.config.part_pnum]), dim=1)
            else:
                if torch.abs(torch.mean(self.fake_latent[:, j*self.config.latent_dim:(j+1)*self.config.latent_dim])) <= self.config.th_select:
                    continue
                else:
                    self.gen_pcd = torch.cat((self.gen_pcd, self.part_dec.module.parts[j](\
                        self.fake_latent[:, self.config.latent_dim*j:self.config.latent_dim*(j+1)])), dim=1)

    def test_model(self, data):
        self.raw_pcd = data['raw_pcd'].to(self.config.device)
        self.raw_mask = data['raw_mask'].to(self.config.device)
        self.raw_id = data['raw_id']
        self.gt_pcd = data['gt_pcd'].to(self.config.device)
        self.partial_pcd = data['partial_pcd'].to(self.config.device)
        
        self.load_ckpts()
        # print(self.raw_id)

        res_path = os.path.join('/'.join(self.config.gan_ckpt_path.split('/')[:-2]), 'results/test', str(self.raw_id[0]))
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with h5py.File(os.path.join(res_path, 'real.h5'), 'w') as f:
            f.create_dataset('pcd', data=self.gt_pcd[0].detach().cpu().numpy())
        
        max_run = 1000
        count = self.config.num_z
        while(count > 0):
            if max_run == 0:
                print(self.raw_id[0])
                shutil.rmtree(res_path)
                break
            max_run -= 1
            self.generate_noise(1)
            with torch.no_grad():
                self.netG.eval()
                self.fake_latent = self.netG(self.z, self.partial_pcd) # 1*512

            self.get_input_gen()
            if self.gen_pcd[0].shape[0] == 0:
                # print(self.raw_id)
                continue
            gen_pcd = self.gen_pcd[0]
            full_pcd = torch.cat((self.input_pcd[0], self.gen_pcd[0]), dim=0)

            # save
            with h5py.File(os.path.join(res_path, 'fake-z'+str(self.config.num_z-count)+'.h5'), 'w') as f:
                f.create_dataset('1024_pcd', data=sample_point_cloud_by_n(gen_pcd.detach().cpu().numpy(), 1024))
                f.create_dataset('2048_pcd', data=sample_point_cloud_by_n(full_pcd.detach().cpu().numpy(), 2048))
            count -= 1

    def visual_model(self, data):
        self.raw_pcd = data['raw_pcd'].to(self.config.device)
        self.raw_mask = data['raw_mask'].to(self.config.device)
        self.raw_id = data['raw_id']
        self.partial_pcd = data['partial_pcd'].to(self.config.device)
        self.gt_pcd = data['gt_pcd'].to(self.config.device)
        
        self.load_ckpts()
        
        for i in range(self.config.num_z):
            self.generate_noise(1)
            with torch.no_grad():
                self.netG.eval()
                self.fake_latent = self.netG(self.z, self.partial_pcd) # 1*512

            self.get_input_gen()

            gen_pcd = self.gen_pcd[0]
            input_pcd = self.input_pcd[0]
            plot_diff_pcds([gen_pcd, input_pcd],
                    vis=self.vis,
                    title=self.raw_id[0]+'_sample_'+str(i),
                    legend=['gen', 'input'],
                    win='sample_pcds')

            self.get_raw_latent()
            self.get_inputD_latent()
            latent = self.inputD_latent[0].reshape(self.config.part_num, self.config.latent_dim)

            self.curve_data = {'X':list(range(self.config.latent_dim)), 'Y':latent.T.tolist(), 'legend':list(range(self.config.part_num))}                    
            self.vis.line(
                X=np.array(self.curve_data['X']),
                Y=np.array(self.curve_data['Y']),
                opts={
                    'title': self.raw_id[0]+'_latent_'+str(i),
                    'legend': ['part_'+str(i) for i in self.curve_data['legend']],
                    'xlabel': 'dim',
                    'ylabel': 'value'},
                win='latent_curves')
            
            time.sleep(0.5)

    def get_mmd_threshold(self, data, ref):
        self.raw_pcd = data['raw_pcd'].to(self.config.device)
        self.raw_mask = data['raw_mask'].to(self.config.device)
        self.raw_id = data['raw_id']
        self.gt_pcd = data['gt_pcd'].to(self.config.device)
        self.partial_pcd = data['partial_pcd'].to(self.config.device)
        self.load_ckpts()
        self.ref = ref.to(self.config.device)
        
        full_pcds = torch.empty(size=(0, 2048, 3)).to(self.config.device)
        count = 100
        while(count > 0):
            self.generate_noise(1)
            with torch.no_grad():
                self.netG.eval()
                self.fake_latent = self.netG(self.z, self.partial_pcd) # 1*512

            self.get_input_gen()
            if self.gen_pcd[0].shape[0] == 0:
                continue

            full_pcd = torch.cat((self.input_pcd[0], self.gen_pcd[0]), dim=0)
            # print(full_pcd.shape)

            full_pcd = torch.from_numpy(sample_point_cloud_by_n(full_pcd.detach().cpu().numpy(), 2048)).unsqueeze(dim=0).to(self.config.device)
            full_pcds = torch.cat((full_pcds, full_pcd), dim=0)
            count -= 1
        
        mmd_list, _ = self.mmd_cd(full_pcds, self.ref)
        min_mmd = mmd_list.min().item()
        max_mmd = mmd_list.max().item()
        res_path = os.path.join('/'.join(self.config.gan_ckpt_path.split('/')[:-2]), 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, 'mmd_threshold.txt'), 'a') as f:
            f.write('%.6f %.6f\n' % (min_mmd, max_mmd))
        
    def get_thmmd(self):
        min_mmd_list = []
        max_mmd_list = []
        with open(os.path.join('/'.join(self.config.gan_ckpt_path.split('/')[:-2]), 'results/mmd_threshold.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                min_mmd, max_mmd = line.split(' ')
                min_mmd_list.append(float(min_mmd))
                max_mmd_list.append(float(max_mmd))

        min_thmmd = sum(min_mmd_list)/len(min_mmd_list)
        max_thmmd = sum(max_mmd_list)/len(max_mmd_list)
        
        print(min_thmmd, max_thmmd)
        return min_mmd_list, min_thmmd, max_thmmd

    def get_tmd(self, data, ref, th):
        self.raw_pcd = data['raw_pcd'].to(self.config.device)
        self.raw_mask = data['raw_mask'].to(self.config.device)
        self.raw_id = data['raw_id']
        self.partial_pcd = data['partial_pcd'].to(self.config.device)

        self.load_ckpts()
        self.ref = ref.to(self.config.device)
        
        gen_pcds = []

        max_run = 100
        count = self.config.num_z
        while (count > 0):
            if max_run == 0:
                print(self.raw_id[0])
                break
            max_run -= 1
            self.generate_noise(1)
            with torch.no_grad():
                self.netG.eval()
                self.fake_latent = self.netG(self.z, self.partial_pcd) # 1*512
            
            self.get_input_gen()
            if self.gen_pcd[0].shape[0] == 0:
                # break
                continue
            
            full_pcd = torch.cat((self.input_pcd[0], self.gen_pcd[0]), dim=0)
            full_pcd = torch.from_numpy(sample_point_cloud_by_n(full_pcd.detach().cpu().numpy(), 2048)).unsqueeze(dim=0).to(self.config.device)
            
            mmd_list, _ = self.mmd_cd(full_pcd, self.ref)
            if mmd_list.item() <= th:
                count -= 1
                gen_pcd = torch.from_numpy(sample_point_cloud_by_n(self.gen_pcd[0].detach().cpu().numpy(), 1024)).unsqueeze(dim=0).to(self.config.device)
                gen_pcds.append(gen_pcd)
            else:
                continue

        tmd = self.tmd(gen_pcds) if count == 0 else 0
        return tmd

    def save_realscan(self, data):
        self.raw_pcd = data['raw_pcd'].to(self.config.device)
        self.raw_mask = data['raw_mask'].to(self.config.device)
        self.raw_id = data['raw_id']
        self.partial_pcd = data['partial_pcd'].to(self.config.device)

        res_path = os.path.join('/'.join(self.config.gan_ckpt_path.split('/')[:-2]), 'results/realscan', self.raw_id[0])
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        input_pcd = torch.from_numpy(sample_point_cloud_by_n(self.raw_pcd[0].detach().cpu().numpy(), 2048*torch.sum(self.raw_mask == 1)//len(self.raw_mask))).float().to(self.config.device)
        
        self.load_ckpts()
        with h5py.File(os.path.join(res_path, 'input.h5'), 'w') as f:
            f.create_dataset('pcd', data=input_pcd.detach().cpu().numpy())
        
        for i in range(self.config.num_z):
            self.generate_noise(1)
            with torch.no_grad():
                self.netG.eval()
                self.fake_latent = self.netG(self.z, self.partial_pcd) # 1*512

            self.get_input_gen()
            # visual
            gen_pcd = self.gen_pcd[0]
            plot_diff_pcds([gen_pcd, input_pcd],
                    vis=self.vis,
                    title='sample'+str(i),
                    legend=['gen', 'input'],
                    win='realscan'+str(i))
            # save
            full_pcd = torch.cat((input_pcd, gen_pcd), dim=0)
            with h5py.File(os.path.join(res_path, 'fake-z'+str(i)+'.h5'), 'w') as f:
                f.create_dataset('pcd', data=full_pcd.detach().cpu().numpy())
    