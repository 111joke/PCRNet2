
from __future__ import print_function
from packaging import version

import torch
from torch import nn
import math
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from dgl.nn.pytorch.factory import KNNGraph
from dgl.nn.pytorch import TAGConv, GATConv, GATv2Conv
import torch.nn.functional as F
import dgl.backend as B
import dgl.function as fn
import dgl
import numpy as np
from torch.nn import init
from VGG import  VGG16

# from pytorch_msssim import SSIM
# from piqa.ssim import SSIM


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hop):
        super(Encoder, self).__init__()
        self.conv1 = TAGConv(in_dim, hidden_dim, k=num_hop)
        self.l2norm = Normalize(2)

    def forward(self, g, edge_weight=None):
        h = g.ndata['h']
        h = self.l2norm(self.conv1(g, h, edge_weight))

        return h

class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=3, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def nonzero_graph(x, th, exist_adj=None):
    if exist_adj is None:
        if B.ndim(x) == 2:
            x = B.unsqueeze(x, 0)
        n_samples, n_points, _ = B.shape(x)

        dist = torch.bmm(x, x.transpose(2, 1)).squeeze().detach()
        base = torch.zeros_like(dist).cuda()
        base[dist > th] = 1
        adj = sparse.csr_matrix(B.asnumpy((base).squeeze(0)))
    else:
        adj = exist_adj
    g = DGLGraph(adj, readonly=True)

    return g, adj

class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        bsz = x.shape[0]

        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

def edge_loss(generated, real, weight=1.0, epsilon=1e-6): # todo 新加入边缘损失
    """
    输入两个(256,256)张量，返回边缘特征损失值

    参数说明:
    generated : torch.Tensor - 生成图像 (256,256)
    real      : torch.Tensor - 真实图像 (256,256)
    weight    : float - 损失权重 (默认1.0)
    epsilon   : float - 数值稳定系数 (默认1e-6)
    """
    # 添加批次和通道维度
    gen = generated.unsqueeze(0).unsqueeze(0)  # (1,1,256,256)
    tgt = real.unsqueeze(0).unsqueeze(0)  # (1,1,256,256)

    # Sobel边缘检测核
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=generated.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=generated.device)

    # 计算梯度
    grad_gen_x = F.conv2d(gen, sobel_x.view(1, 1, 3, 3), padding=1)
    grad_gen_y = F.conv2d(gen, sobel_y.view(1, 1, 3, 3), padding=1)
    grad_tgt_x = F.conv2d(tgt, sobel_x.view(1, 1, 3, 3), padding=1)
    grad_tgt_y = F.conv2d(tgt, sobel_y.view(1, 1, 3, 3), padding=1)

    # 边缘强度计算
    edge_gen = torch.sqrt(grad_gen_x.pow(2) + grad_gen_y.pow(2) + epsilon)
    edge_tgt = torch.sqrt(grad_tgt_x.pow(2) + grad_tgt_y.pow(2) + epsilon)

    # 复合损失计算
    l1_loss = F.l1_loss(edge_gen, edge_tgt)
    ssim_loss = 1 - torch.mean(torch.exp(-torch.abs(edge_gen - edge_tgt)))

    # 移除额外维度后返回
    return weight * (l1_loss + 0.5 * ssim_loss).squeeze()

#todo 新加入
class PaCoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, K=128,
                 num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features_q, features_k, labels=None, sup_logits=None):
        device = torch.device('cuda:0' if features_q.is_cuda else 'cpu')

        # Since we are dealing with a single sample, batch_size is 1
        batch_size = 1

        # Flatten the spatial dimensions for both feature tensors
        features_q = features_q.view(1, -1)  # Shape: (1, H*W)
        features_k = features_k.view(1, -1)  # Shape: (1, H*W)

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels[:batch_size], labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        # Compute logits using complete features tensor
        anchor_dot_contrast = torch.div(
            torch.matmul(features_q, features_k.T),
            self.temperature)

        # Add supervised logits
        if sup_logits is not None:
            sup_logits = sup_logits.to(device)
            anchor_dot_contrast = torch.cat(((sup_logits) / self.supt, anchor_dot_contrast), dim=1)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # Add ground truth
        if labels is not None:
            one_hot_label = F.one_hot(labels[:batch_size, ].view(-1, ), num_classes=self.num_classes).to(torch.float32)
            one_hot_label = one_hot_label.to(device)
            mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # Compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
class GNNLoss(nn.Module):
    def __init__(self, opt, use_mlp=True):
        super(GNNLoss, self).__init__()

        ## graph arguments
        self.num_hop = 2
        self.pooling_num = 1
        self.down_scale = 8
        # self.pooling_ratio = opt.pooling_ratio
        self.nonzero_th = 0.6

        self.gpu_ids = [0]
        self.nc = 256
        self.num_patch = 256
        self.init_type = 'normal'
        self.init_gain = 0.02

        self.use_mlp = use_mlp
        self.mlp_init = False

        # self.criterion = NCESoftmaxLoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.normalization = Normalization(self.device)
        self.netPre = VGG16().to(self.device)
        self.criterion = NCESoftmaxLoss()

        self.optattn_layers = "4,7,9"
        self.attn_layers = [int(i) for i in self.optattn_layers.split(',')]
        # todo 新加入PaColoss损失
        self.block = PaCoLoss()


    def create_mlp(self, feats):  # 创建多层感知机（MLP）
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            embed = Embed(input_nc, self.nc)
            pools = nn.ModuleList()
            gnn_pools = nn.ModuleList()
            gnn = Encoder(self.nc, self.nc, self.num_hop)

            if len(self.gpu_ids) > 0:
                gnn.cuda()
                pools.cuda()
                gnn_pools.cuda()
                embed.cuda()

            setattr(self, 'gnn_%d' % mlp_id, gnn)
            setattr(self, 'embed_%d' % mlp_id, embed)
            setattr(self, 'pools_%d' % mlp_id, pools)
            setattr(self, 'gnn_pools_%d' % mlp_id, gnn_pools)

        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def calc_gnn(self, f_es, f_et, num_patches, gnn=None, adj_s=None, adj_t=None): # f_es生成图
        batch_dim_for_bmm = 1
        T = 0.07
        ## input features
        G_pos_t, adj_t = nonzero_graph(f_et.detach(), self.nonzero_th, exist_adj=adj_t)
        G_pos_t = G_pos_t.to(f_es.device)
        G_pos_t = dgl.add_self_loop(G_pos_t)
        G_pos_t.ndata['h'] = f_et
        f_gt = gnn(G_pos_t)
        f_gt = f_gt.detach()

        ## output features
        G_pos_s, adj_s = nonzero_graph(f_es.detach(), self.nonzero_th, exist_adj=adj_t)
        G_pos_s = G_pos_s.to(f_es.device)
        G_pos_s = dgl.add_self_loop(G_pos_s)
        G_pos_s.ndata['h'] = f_es
        f_gs = gnn(G_pos_s)  ## shared GNN

        ## node-wise contrastive loss
        f_gt = f_gt.squeeze()
        f_gs = f_gs.squeeze()
        gs_pos = torch.einsum('nc,nc->n', [f_gt, f_gs]).unsqueeze(-1)
        gt_pos = torch.einsum('nc,nc->n', [f_gs, f_gt]).unsqueeze(-1)

        f_gt_reshape = f_gt.view(batch_dim_for_bmm, -1, self.nc)
        f_gs_reshape = f_gs.view(batch_dim_for_bmm, -1, self.nc)
        gs_neg = torch.bmm(f_gt_reshape, f_gs_reshape.transpose(2, 1))
        gt_neg = torch.bmm(f_gs_reshape, f_gt_reshape.transpose(2, 1))  # .squeeze()

        diagonal = torch.eye(num_patches, device=f_es.device, dtype=torch.bool)[None, :, :]
        gs_neg.masked_fill_(diagonal, -10.0)
        gt_neg.masked_fill_(diagonal, -10.0)

        gs_neg = gs_neg.view(-1, num_patches)
        gt_neg = gt_neg.view(-1, num_patches)

        out_gs = torch.cat([gs_pos, gs_neg], dim=1)
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()
        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()

        loss_gs = self.criterion(out_gs)
        loss_gt = self.criterion(out_gt)
        loss_g = loss_gs + loss_gt

        return loss_g, f_gs, f_gt, adj_s, adj_t

    def forward(self, feat_s, feat_t, num_patches=64):  # 第一个是生成图，第二个是真实图
        ## get feat
        norm_real_A, norm_fake_B = self.normalization((feat_t + 1) * 0.5), \
                                                 self.normalization((feat_s + 1) * 0.5)
        feat_s = self.netPre(norm_fake_B, self.attn_layers, encode_only=True)
        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     fake_B_feat = [torch.flip(fq, [3]) for fq in fake_B_feat]
        feat_t = self.netPre(norm_real_A, self.attn_layers, encode_only=True) # 由于提取了三层，每一层都是（2，3，256，256），其中batchsize为1，所以提取到的是一个列表

        loss_g_total = 0
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feat_s)
        for mlp_id, (fs, ft) in enumerate(zip(feat_s, feat_t)):
            loss_g = 0
            gnn = getattr(self, 'gnn_%d' % mlp_id)
            embed = getattr(self, 'embed_%d' % mlp_id)
            pools = getattr(self, 'pools_%d' % mlp_id)
            gnn_pools = getattr(self, 'gnn_pools_%d' % mlp_id)

            if fs.dim() == 3:  # 如果是 3D 张量，添加 batch_size 维度
                fs = fs.unsqueeze(0)  # 变成 (1, 3, 256, 256)
            fs_reshape = fs.permute(0, 2, 3, 1).flatten(1, 2)  # fs为生成图像，ft为原始图像
            if ft.dim() == 3:  # 如果是 3D 张量，添加 batch_size 维度
                ft = ft.unsqueeze(0)  # 变成 (1, 3, 256, 256)
            ft_reshape = ft.permute(0, 2, 3, 1).flatten(1, 2)

            if num_patches > 0:

                patch_id = torch.randperm(fs_reshape.shape[1], device=fs[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                # fs_sample = fs_reshape[:, patch_id, :].flatten(0, 1) # fs_sample为生成图，ft_sample为原始图
                # ft_sample = ft_reshape[:, patch_id, :].flatten(0, 1)

                # 合并 batch_size 中的所有补丁，并保留 num_patches_selected 和 feature_dim 维度
                # 合并 batch_size 中的所有补丁，并保留 num_patches_selected 和 feature_dim 维度
                fs_sample = fs_reshape[:, patch_id, :].mean(dim=0)  # 生成图
                ft_sample = ft_reshape[:, patch_id, :].mean(dim=0)  # 原始图

            else:
                fs_sample = fs_reshape
                ft_sample = ft_reshape

            lossedge = edge_loss(fs_sample,ft_sample) # todo 新添加边缘损失


            ### embed pretrained_enc feature by mlp
            # ft_sample = ft_sample.view(-1, 256)
            # fs_sample = fs_sample.view(-1, 256)
            f_et_embed = embed(ft_sample) #todo f_et_embed为原始图，f_es_embed为生成图
            f_es_embed = embed(fs_sample)
            # todo 这里加入PaColoss损失
            labels = torch.randint(0, 10, (1,))  # Example label for a single sample
            sup_logits = torch.rand(1, 1000)
            lossPaco = self.block(f_es_embed, f_et_embed, labels=labels, sup_logits=sup_logits)

            ### graph loss before pooling
            loss_gnn, gnn_s, gnn_t, adj_s, adj_t = self.calc_gnn(f_es_embed, f_et_embed,
                                                                 num_patches, gnn=gnn)

            return loss_gnn + lossedge + lossPaco




