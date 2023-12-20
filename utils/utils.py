import numpy as np
import random
import torch
import json
from plyfile import PlyData, PlyElement
from torch.autograd import grad
from matplotlib import cm


def downsample_point_cloud(points, n_pts):
    """downsample points by random choice

    :param points: (n, 3)
    :param n_pts: int
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts)
    return points[p_idx]

def upsample_point_cloud(points, n_pts):
    """upsample points by random choice

    :param points: (n, 3)
    :param n_pts: int, > n
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts - points.shape[0])
    dup_points = points[p_idx]
    points = np.concatenate([points, dup_points], axis=0)
    return points

def sample_point_cloud_by_n(points, n_pts):
    """resample point cloud to given number of points"""
    if n_pts > points.shape[0]:
        return upsample_point_cloud(points, n_pts)
    elif n_pts < points.shape[0]:
        return downsample_point_cloud(points, n_pts)
    else:
        return points

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

def part_normalization(pcd, part_num):
    """here only consider translate
    
    :param pcd: (B, pnum, 3)
    :param part_num:
    :return: 
        normed_pcd: (B, pnum, 3)
        trans: (B, part_num, trans_dim)
    """
    pnum = pcd.shape[1]
    pdim = pcd.shape[2]

    pcd = pcd.reshape(-1, part_num, pnum//part_num, pdim)
    trans = torch.mean(pcd, dim=2, keepdim=True)

    normed_pcd = (pcd - trans).reshape(-1, pnum, pdim)
    trans = trans.squeeze(dim=2)
    return normed_pcd, trans

def plot_diff_pcds(pcds, vis, title, legend, win=None):
    """
    :param pcds: python list, include pcds with different size
    :      legend: each pcds' legend
    :return:
    """
    device = pcds[0].device
    assert vis.check_connection()

    pcds_data = torch.Tensor().to(device)
    for i in range(len(pcds)):
        pcds_data = torch.cat((pcds_data, pcds[i][:,[2,0,1]]), 0) # partnet

    pcds_label = torch.Tensor().to(device)
    for i in range(1, len(pcds) + 1):
        pcds_label = torch.cat((pcds_label, torch.Tensor([i] * pcds[i - 1].shape[0]).to(device)), 0)

    vis.scatter(X=pcds_data, Y=pcds_label,
                opts={
                    'title': title,
                    'markersize': 2,
                    # 'markercolor': np.random.randint(0, 255, (len(pcds), 3)),
                    'webgl': True,
                    'legend': legend},
                win=win)

def generate_cmap(color_num, cmap='viridis'):
    """
    https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
    """
    viridis = cm.get_cmap(cmap)
    rgb = np.trunc(viridis(np.linspace(0, 1, color_num))[:,0:3]*255)
    return rgb  # color_num * 3

def plot_diff_regions(pcd, vis, title, labels, win=None):
    '''
    :param pcd: one pcd, diff color for diff region
    :return:
    '''
    device = pcd.device
    assert vis.check_connection()

    pcd_data = pcd[:,[2,0,1]]
    pcd_label = labels.int()
    region_num = torch.unique(labels).shape[0]

    vis.scatter(X=pcd_data, Y=pcd_label,
                # update='append',
                opts={
                    'title': title,
                    'markersize': 2,
                    'markercolor': generate_cmap(region_num),
                    'webgl': True,
                    'legend': ['region'+str(i) for i in range(region_num)]},
                win=win)

class Config():
    def __init__(self):
        args = self.parse()

        # set as attributes
        for k, v in args.items():
            # print(k, v)
            self.__setattr__(k, v)

    def parse(self):
        with open('config.json', 'r') as f:
            data = json.load(f)
        
        args = {}
        args['task'] = data['task']
        args['vis'] = data['vis']
        args['gpu_ids'] = data['gpu_ids']
        torch.cuda.set_device('cuda:'+str(args['gpu_ids'][0]))
        args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        args.update(data['configs']['base'])
        args.update(data['configs']['adjust'])
        return args

# taken from https://github.com/SymenYang/CPCGAN/blob/main/Model/Gradient_penalty.py
class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=torch.device('cpu')):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.device = device

    def __call__(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        
        fake_data = fake_data[:batch_size]
        
        # alpha = torch.rand(batch_size, 1, requires_grad=True).to(self.device)
        alpha = torch.rand_like(real_data, requires_grad=True).to(self.device)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates)
        # compute gradients w.r.t the interpolated outputs
        
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size,-1)
                         
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty
