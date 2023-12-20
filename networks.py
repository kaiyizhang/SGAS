import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################################
# PartAE
#############################################################

class EncoderPointNet(nn.Module):
    def __init__(self, n_filters, latent_dim, bn=True):
        super(EncoderPointNet, self).__init__()
        self.n_filters = list(n_filters) + [latent_dim]
        self.latent_dim = latent_dim

        model = []
        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.model(x)
        x = torch.max(x, dim=2)[0]
        return x


class DecoderFC(nn.Module):
    def __init__(self, n_features, latent_dim, output_pts, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], self.output_pts*3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        x = torch.transpose(x, 1, 2)
        return x


# Only one branch is activated for each training
class PartAE(nn.Module):
    def __init__(self, config):
        super(PartAE, self).__init__()
        self.config = config

        self.parts = nn.ModuleList([nn.Sequential(
                        EncoderPointNet(config.enc_pnfilters, config.latent_dim), 
                        DecoderFC(config.dec_features, config.latent_dim, config.part_pnum)
                        ) for i in range(config.part_num)])

    def forward(self, x): # x = one part
        return self.parts[self.config.part_id](x)


class Pcd2Feat(nn.Module):
    def __init__(self, config):
        super(Pcd2Feat, self).__init__()
        self.config = config

        self.parts = nn.ModuleList([
                        EncoderPointNet(config.enc_pnfilters, config.latent_dim)
                        for i in range(config.part_num)])

    def forward(self, x): # x = one part
        return self.parts[self.config.part_id](x)


class Feat2Pcd(nn.Module):
    def __init__(self, config):
        super(Feat2Pcd, self).__init__()
        self.config = config

        self.parts = nn.ModuleList([
                        DecoderFC(config.dec_features, config.latent_dim, config.part_pnum)
                        for i in range(config.part_num)])

    def forward(self, x_feat):
        return self.parts[self.config.part_id](x_feat)


#############################################################
# SGAS
#############################################################

class GeneratorPart(nn.Module):
    def __init__(self, n_features, noise_dim, latent_dim, bn=True):
        super(GeneratorPart, self).__init__()
        self.n_features = list(n_features)
        self.noise_dim = noise_dim

        model = []
        prev_nf = self.noise_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], latent_dim)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class GeneratorComp(nn.Module):
    def __init__(self, config):
        super(GeneratorComp, self).__init__()
        self.config = config

        self.parts = nn.ModuleList([
                        GeneratorPart(config.partG_features, config.noise_dim, config.latent_dim)
                        for i in range(config.part_num)])
        self.encoder = StyleEncoder()

    def forward(self, x, x_p): # B*128, B*1024*3
        mu, var = self.encoder(x_p)
        # mu = 0
        # var = 1

        x = mu + x * var
        # print(x.shape)

        x_latent = torch.empty(size=(x.shape[0], 0)).to(x.device)
        for i in range(self.config.part_num):
            # print(self.parts[i](x).shape)
            x_latent = torch.cat((x_latent, self.parts[i](x)), dim=1)
        return x_latent


class GeneratorGen(nn.Module):
    def __init__(self, config):
        super(GeneratorGen, self).__init__()
        self.config = config

        self.parts = nn.ModuleList([
                        GeneratorPart(config.partG_features, config.noise_dim, config.latent_dim)
                        for i in range(config.part_num)])

    def forward(self, x): # B*128
        x_latent = torch.empty(size=(x.shape[0], 0)).to(x.device)
        for i in range(self.config.part_num):
            # print(self.parts[i](x).shape)
            x_latent = torch.cat((x_latent, self.parts[i](x)), dim=1)
        return x_latent


class DiscriminatorPart(nn.Module):
    def __init__(self, latent_dim, n_features, bn=False):
        super(DiscriminatorPart, self).__init__()
        self.n_features = list(n_features)
        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf
        model.append(nn.Linear(self.n_features[-1], 1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        # x = self.model(x).view(-1)
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.parts = nn.ModuleList([
                        DiscriminatorPart(config.latent_dim, config.partD_features) 
                        for i in range(config.part_num)])

    def forward(self, x): # B*(128*self.part_num)
        x_value = torch.empty(size=(x.shape[0], 0)).to(x.device)
        for i in range(self.config.part_num):
            x_value = torch.cat((x_value, \
                self.parts[i](x[:, self.config.latent_dim*i:self.config.latent_dim*(i+1)])), dim=1)
        return x_value


class DiscriminatorFull(nn.Module):
    def __init__(self, config):
        latent_dim = config.latent_dim*config.part_num
        n_features = config.partD_features
        bn = False

        super(DiscriminatorFull, self).__init__()
        self.n_features = list(n_features)
        model = []
        prev_nf = latent_dim
        for idx, nf in enumerate(self.n_features):
            model.append(nn.Linear(prev_nf, nf))
            model.append(nn.LeakyReLU(inplace=True))
            prev_nf = nf
        model.append(nn.Linear(self.n_features[-1], 1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x).view(-1)
        return x


#############################################################
# StyleEncoder
#############################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_filters=(64, 128, 128, 256), latent_dim=128, z_dim=128, bn=True):
        super(StyleEncoder, self).__init__()
        self.n_filters = list(n_filters) + [latent_dim]
        self.latent_dim = latent_dim
        self.z_dim = z_dim

        model = []
        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.ReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)

        self.fc_mu = nn.Linear(latent_dim, z_dim)
        self.fc_var = nn.Linear(latent_dim, z_dim)

    def forward(self, x):
        # print(x.shape)
        x = torch.transpose(x, 1, 2)
        x = self.model(x)
        x = torch.max(x, dim=2)[0]

        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var


#############################################################
# StructGen
#############################################################

class PointNet(nn.Module):
    def __init__(self, K1):
        super(PointNet, self).__init__()
        self.K1 = K1

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.K1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.K1)
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, dim=2)
        return x


class AXform(nn.Module):
    def __init__(self, K1, K2, N):
        super(AXform, self).__init__()
        self.K1 = K1
        self.K2 = K2
        self.N = N  # N>=K2

        self.fc1 = nn.Linear(K1, N*K2)

        self.conv1 = nn.Conv1d(K2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=2)

        self.conv4 = nn.Conv1d(K2, 3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.N, self.K2)

        x_base = x
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x_weights = self.softmax(x)
        x = torch.bmm(x_weights, x_base)

        x = x.transpose(1, 2).contiguous()
        x = self.conv4(x)
        x = x.transpose(1, 2).contiguous()
        return x


class StructGen(nn.Module):
    def __init__(self, config):
        super(StructGen, self).__init__()
        self.pointnet = PointNet(K1=config.hparas[0])
        self.decoder = nn.ModuleList([AXform(K1=config.hparas[0], K2=config.hparas[1], N=config.hparas[2]) \
            for i in range(config.part_num)])

    def forward(self, x):
        x_feat = self.pointnet(x)

        x = torch.empty(size=(x.shape[0], 0, 3)).to(x.device)
        for i, data in enumerate(self.decoder):
            _x = self.decoder[i](x_feat)
            x = torch.cat((x, _x), dim=1)
        return x  # B*N*3
