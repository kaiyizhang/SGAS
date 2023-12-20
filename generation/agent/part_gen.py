import os
import torch
import torch.nn as nn
from visdom import Visdom

import sys 
sys.path.append("..")

from agent.part_ae import PartAEAgent
from utils.utils import *
from networks import *

from utils.metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from utils.metrics.evaluation_metrics import compute_all_metrics


class PartGenAgent(PartAEAgent):
    def __init__(self, config):
        self.config = config
        config.part_pnum = int(2048//config.part_num*(1+self.config.ext_ratio))
        self.load_pretrained_AE()
        self.build_net()

        if self.config.task == 'train':
            self.vis = Visdom(env=config.visdom_env)
            self.init_log()
            self.set_optimizer()
            self.loss_gp = GradientPenalty(lambdaGP=10, gamma=1, device=config.device)
            self.best_jsd = 0
        else:
            self.vis = Visdom(env=config.visdom_env)
            self.metrics = {}
            self.fpd = 0

    def build_net(self):
        self.netG = nn.DataParallel(
            GeneratorGen(self.config).to(self.config.device), device_ids=self.config.gpu_ids)
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

    def train_model(self, data, args):
        self.real_pcd = data['real_pcd'].to(self.config.device)
        self.real_id = data['real_id']
        self.gt_pcd = data['gt_pcd'].to(self.config.device)

        self.get_real_latent()
        self.generate_noise(self.real_pcd.shape[0])

        # update D network
        self.fake_latent = self.netG(self.z)

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
        fake_out_full = self.netD_full(self.fake_latent.detach())
        self.loss_D_real_full = -torch.mean(real_out_full)
        self.loss_D_fake_full = torch.mean(fake_out_full)
        self.loss_D_gp_full = self.loss_gp(self.netD_full, self.real_latent, self.fake_latent.detach())
        self.loss_D_full = self.loss_D_real_full + self.loss_D_fake_full + self.loss_D_gp_full
        
        self.optimizerD_full.zero_grad()
        self.loss_D_full.backward()
        self.optimizerD_full.step()

        if args[1] % 5 - 1 == 0:
            # update G network
            self.fake_latent = self.netG(self.z)

            fake_out = self.netD(self.fake_latent)
            full_out = self.netD_full(self.fake_latent)
            self.loss_G = self.config.g_ratio[0] * -torch.mean(fake_out)
            self.loss_G_full = self.config.g_ratio[1] * -torch.mean(full_out)

            self.optimizerG.zero_grad()
            (self.loss_G + self.loss_G_full).backward()
            self.optimizerG.step()

    def val_model(self, ref_pcs):
        sample_pcs = []
        for i in range(ref_pcs.shape[0]):
            _, out_pc = self.inference()
            sample_pcs.append(out_pc)
        sample_pcs = torch.cat(sample_pcs, dim=0)

        self.jsd = JSD(sample_pcs, ref_pcs)
        print('jsd:', self.jsd)

    def save_ckpts(self, step, ref_pcs):
        save_path = os.path.join(self.config.save_path, 'ckpts')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.epoch % step == 0:
            torch.save({'epoch': self.epoch, 'state_dict': self.netG.state_dict()}, os.path.join(save_path, 'epoch_%d.pth' % (self.epoch)))
        if self.epoch % 50 == 0:
            self.val_model(ref_pcs)
            if self.epoch == 50 or self.jsd <= self.best_jsd:
                self.best_jsd = self.jsd
                torch.save({'epoch': self.epoch, 'state_dict': self.netG.state_dict()}, os.path.join(save_path, 'epoch_best.pth'))

    def after_one_epoch(self, args, ref_pcs):
        self.epoch = args[0]

        fake_pcd = torch.empty(size=(self.fake_latent.shape[0], 0, 3)).to(self.fake_latent.device)
        for i in range(self.config.part_num):
            fake_pcd = torch.cat((fake_pcd, self.part_dec.module.parts[i](\
                self.fake_latent[:, self.config.latent_dim*i:self.config.latent_dim*(i+1)])), dim=1)

        recon_pcd = torch.empty(size=(self.real_latent.shape[0], 0, 3)).to(self.real_latent.device)
        for i in range(self.config.part_num):
            recon_pcd = torch.cat((recon_pcd, self.part_dec.module.parts[i](\
                self.real_latent[:, self.config.latent_dim*i:self.config.latent_dim*(i+1)])), dim=1)

        if self.config.vis:
            plot_diff_pcds([fake_pcd[0], self.real_pcd[0], recon_pcd[0]],
                                vis=self.vis,
                                title='epoch %d' % self.epoch,
                                legend=['fake', 'real', 'recon'],
                                win='visual')

        losses = {'loss_G': self.loss_G.item(),
                  'loss_G_full': self.loss_G_full.item(),
                  'loss_D': self.loss_D.item(),
                  'loss_D_full': self.loss_D_full.item()}
        self.loss_curves(losses)

        self.save_ckpts(step=self.config.save_frequency, ref_pcs=ref_pcs)

    def load_ckpts(self):
        ckpt = torch.load(self.config.gan_ckpt_path, map_location=lambda storage, location: storage)
        print("load epoch: %d" % (ckpt['epoch']))
        self.netG.load_state_dict(ckpt['state_dict'])

    def inference(self):
        self.generate_noise(1)
        with torch.no_grad():
            self.netG.eval()
            self.fake_latent = self.netG(self.z)
        fake_pcd = torch.empty(size=(self.fake_latent.shape[0], 0, 3)).to(self.fake_latent.device)
        for i in range(self.config.part_num):
            fake_pcd = torch.cat((fake_pcd, self.part_dec.module.parts[i](\
                self.fake_latent[:, self.config.latent_dim*i:self.config.latent_dim*(i+1)])), dim=1)
        fake_pcd = fake_pcd.squeeze()
        # print(fake_pcd.shape)

        out_pc = torch.from_numpy(sample_point_cloud_by_n(fake_pcd.detach().cpu().numpy(), \
                        2048)).unsqueeze(0).to(self.config.device)
        return fake_pcd, out_pc  # 1*2048*3

    def visual_model(self, data):
        te_pc = data['gt_pcd'].to(self.config.device)[0]
        self.load_ckpts()
        _, out_pc = self.inference()
        print(te_pc.shape)
        print(out_pc.shape)
        plot_diff_pcds([out_pc.squeeze(), te_pc.squeeze()],
                            vis=self.vis,
                            title=data['real_id'][0],
                            legend=['fake', 'real'],
                            win=data['real_id'][0])

    def calc_metrics(self, ref_pcs):
        ref_pcs = ref_pcs[torch.randperm(ref_pcs.shape[0])[:100]]  # random get 100 ref pcds
        self.load_ckpts()
        
        sample_pcs = []
        for i in range(150):  # random get 150 sample pcds
            _, out_pc = self.inference()
            sample_pcs.append(out_pc)
        sample_pcs = torch.cat(sample_pcs, dim=0)

        print("Generation sample size:%s reference size: %s"
            % (sample_pcs.size(), ref_pcs.size()))
        
        # Compute metrics
        results = compute_all_metrics(sample_pcs, ref_pcs, batch_size=100, accelerated_cd=True)
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in results.items()}
        results = {k: results[k] for k in sorted(results)}
        for k, v in results.items():
            print(k+':', v)

        for key in results:
            if key in self.metrics:
                self.metrics[key] += results[key]
            else:
                self.metrics[key] = results[key]
