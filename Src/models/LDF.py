# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：LDF baseline model
# Author: Yinda XU

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from .ResNeXt_DenseNet import resnet_mod56, resnet50, resnet18
from .SeNet import se_resnet

class LDF_baseline(nn.Module):
    def __init__(
        self, arr_ClassEmbed, dimVisFeat,
        NormalizedLogit=False, RetLogits = False,
    ):
        super(LDF_baseline, self).__init__()
        num_classes, dim_ClassEmbed = arr_ClassEmbed.shape
        self.NormalizedLogit = NormalizedLogit
        self.RetLogits = RetLogits

        # self.ConvNet = resnet18(num_classes=dim_ClassEmbed)
        self.ConvNet = se_resnet(num_classes=dim_ClassEmbed)
        ts_ClassEmbed_t=Tensor(arr_ClassEmbed.T)
        if self.NormalizedLogit:
            ts_ClassEmbed_t=ts_ClassEmbed_t.renorm(p=2, dim=0, maxnorm=1)
        self.ClassEmbed_t = Parameter(ts_ClassEmbed_t)
        self.ClassEmbed_t.requires_grad = False
    def forward(self, X, ):
        X_emb = self.ConvNet(X)
        if self.NormalizedLogit:
            X_emb = X_emb.renorm(p=2, dim=0, maxnorm=1)
        scores_emb = torch.matmul(X_emb, self.ClassEmbed_t)
        if self.RetLogits:
            return scores_emb, X_emb
        else:
            return scores_emb

# class LDF_wt_CoarseClass(nn.Module):
#     def __init__(self, model_LDF, model_CoarseClass, sr_LabelEnc, sr_LabelEnc_CoarseClasses):
#         super(LDF_wt_CoarseClass, self).__init__()
#
#         pass
#     def forward(self,x):
#         pass
