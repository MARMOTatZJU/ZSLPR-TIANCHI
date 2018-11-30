import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .hard_triplet_loss import HardTripletLoss

gamma_dft = 20

class loss_QFSL(nn.Module):
    def __init__(self, idxUnseen, gamma=gamma_dft, eps=1e-8):
        super(loss_QFSL, self).__init__()
        self.idxUnseen = idxUnseen
        self.NUnseen = idxUnseen.size
        self.gamma=gamma
        self.eps=eps
    def forward(self, outputs, targetsIdx):
        loss_CE = F.cross_entropy(outputs, targetsIdx, reduce=False)
        proba = F.softmax(outputs, dim=1)
        reg_QFSL = (-torch.log(self.eps+
            proba[:, self.idxUnseen].sum(dim=1)
        )/self.NUnseen)
        loss = (loss_CE + self.gamma*reg_QFSL).mean(dim=0)
        return loss

class loss_SupvisVisFeat(nn.Module):
    def __init__(self, coeff_SupVisFeat, ):
        super(loss_SupvisVisFeat, self).__init__()
        self.coeff_SupVisFeat=coeff_SupVisFeat
    def forward(self, outputs, targetsIdx):
        outputs_emb, outputs_vis = outputs
        loss_emb = F.cross_entropy(outputs_emb, targetsIdx)
        loss_vis = F.cross_entropy(outputs_vis, targetsIdx)
        return  loss_emb + self.coeff_SupVisFeat*loss_vis

class loss_LDF_CE_TriHard(nn.Module):
    def __init__(self, margin = 0.1, coeff_TriHard=0.1):
        super(loss_LDF_CE_TriHard, self).__init__()
        self.coeff_TriHard=coeff_TriHard
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.HardTripletLoss = HardTripletLoss(
            margin=margin, hardest=True, squared=True)
    def forward(self, outputs, targetsIdx):
        scores_emb, X_emb = outputs
        loss_CE = self.CrossEntropyLoss(scores_emb, targetsIdx)
        loss_TriHard = self.HardTripletLoss(X_emb, targetsIdx)
        loss =  loss_CE + self.coeff_TriHard * loss_TriHard
        return loss, loss_TriHard.item()

class loss_fWGAN(nn.Module):

    def __init__(self, G, D, ConvNet, lambda_=10, beta=0.01):
        super(loss_fWGAN, self).__init__()
        self.G = G
        self.D = D
        self.ConvNet = ConvNet
        self.lambda_ = lambda_
        self.beta = beta
    def forward(self, *args, G=False, D=False):
        # generator loss
        if G:
            feat_g, class_emb, targetsIdx = args
            feat_g_valid = self.D(feat_g, class_emb)
            outputs=self.ConvNet(feat_g, MakeInference=True)
            loss_cls = F.cross_entropy(outputs, targetsIdx.squeeze(dim=1), reduce=False )
            loss_g = (
                -feat_g_valid + self.beta*loss_cls
            ).mean(dim=0)
            return loss_g, feat_g_valid, loss_cls
        # discriminator loss
        elif D:
            feat_g, feat_r, class_emb = args
            batch_size, feat_size = feat_g.shape
            feat_g_valid, feat_r_valid = self.D(feat_g, class_emb), self.D(feat_r, class_emb)
            # GP compute
            alpha = torch.rand(batch_size, feat_size).cuda()
            feat_interp = feat_g*alpha+feat_r*(1-alpha)
            feat_interp_valid = self.D(feat_interp, class_emb)
            if torch.__version__>'0.4.0':
                grad_feat_interp_valid = torch.ones(batch_size).cuda()
            else:
                grad_feat_interp_valid = torch.ones(batch_size, 1).cuda()
            grads = torch.autograd.grad(
                outputs=feat_interp_valid,
                inputs=feat_interp,
                grad_outputs=grad_feat_interp_valid,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0].view(batch_size, -1)
            GP = (grads.norm(dim=1)-1)**2
            # total loss
            loss_d = (
                feat_g_valid - feat_r_valid + self.lambda_*GP
            ).mean(dim=0)
            return loss_d, feat_g_valid, feat_r_valid, GP
        else:
            raise RuntimeError('loss_fWGAN error')
