# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：implementation of GAN model
# Author: Yinda XU

import torch
from torch import Tensor
from torch import nn

negative_slope = 0.1
def makeLayers(lst_dim):
    lst_dimInpFC = lst_dim[:-1]
    lst_dimOupFC = lst_dim[1:]
    lst_fc = []
    for ith, (dimInpFC, dimOupFC) in enumerate(zip(lst_dimInpFC, lst_dimOupFC)):
        lst_fc.append(nn.Linear(dimInpFC, dimOupFC))
        if ith == len(lst_dimInpFC)-1:
            lst_fc.append(nn.ReLU())
        else:
            lst_fc.append(nn.LeakyReLU(negative_slope))
    lays = nn.Sequential(*tuple(lst_fc))
    return lays


class G_Feat(nn.Module):
    def __init__(self, emb_size=[300], dimHid=[1024, 4096, ], feat_size=2048, ):
        super(G_Feat, self).__init__()
        self.lst_emb_size = emb_size
        self.feat_size = feat_size
        self.NEmbs = len(emb_size)
        self.lst_layInp = nn.ModuleList([
            # nn.Linear(dimInp*2, dimHid[0]//self.NEmbs)
            nn.Sequential(
                nn.Linear(dimInp*2, dimHid[0]),
                nn.LeakyReLU(negative_slope), )
            for dimInp in self.lst_emb_size], )
        self.lays = makeLayers(dimHid+[feat_size])
    def forward(self, lst_ClassEmb):
        batch_size, _ = lst_ClassEmb[0].shape
        # Transform & concat embeddings
        lst_transformed = []
        for layInp, ClassEmb, emb_size in zip(self.lst_layInp, lst_ClassEmb, self.lst_emb_size):
            Z = torch.randn(batch_size, emb_size).cuda()
            X = torch.cat([Z, ClassEmb], dim=1)
            X = layInp(X)
            lst_transformed.append(X)
        # X = torch.cat(lst_transformed, dim=1)
        X = torch.stack(lst_transformed, dim=2)
        X,_ = X.max(dim=2)
        # Generate features
        X = self.lays(X)
        return X.squeeze(dim=1)


class D_EM(nn.Module):
    def __init__(self, feat_size=2048, emb_size=[300], dimHid=[512, 128], ):
        super(D_EM, self).__init__()
        self.feat_size = feat_size
        self.lst_emb_size = emb_size
        self.NEmbs = len(emb_size)
        self.lst_layInp = nn.ModuleList([
            # nn.Linear(feat_size+dimInp, dimHid[0]//self.NEmbs)
            nn.Sequential(
                nn.Linear(feat_size+dimInp, dimHid[0]),
                nn.LeakyReLU(negative_slope), )
            for dimInp in self.lst_emb_size], )
        self.lays = makeLayers(dimHid+[1])
    def forward(self, feat, lst_ClassEmb):
        # Transform & concat features
        lst_transformed = []
        for layInp, ClassEmb, in zip(self.lst_layInp, lst_ClassEmb, ):
            X = torch.cat([feat, ClassEmb], dim=1)
            X = layInp(X)
            lst_transformed.append(X)
        # X = torch.cat(lst_transformed, dim=1)
        X = torch.stack(lst_transformed, dim=2)
        X,_ = X.max(dim=2)
        X = self.lays(X)
        return X.squeeze(dim=1)

class fWGAN_classifier(nn.Module):
    def __init__(self, dimFeat, NClass):
        super(fWGAN_classifier, self).__init__()
        self.fc = nn.Linear(dimFeat, NClass)
    def forward(self, X):
        X = self.fc(X)
        return X
