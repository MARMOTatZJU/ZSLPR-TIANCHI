# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：run training of fWGAN
# Author: Yinda XU

# import installed packages
import os
import shutil
import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor
from torch import LongTensor
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
# import in-house packages
from dataset import datasetZSL, datasetZSL_wtDataAug
import models
from models import resnet50, G_Feat, D_EM
from models.loss import loss_fWGAN

# import config
import config
from config import *

import utils
from utils.wheels import *
from utils.trainer import trainer_epoch_fWGAN
from utils.DataAug import data_aug_ZSL
# Make experiment result folder
TimeStamp, path_folderExp, path_folderExpBestModel = utils.backup.createBackUp()

from prepareData import *

# Make dataset
dataset_trn = datasetZSL(
    path_folderTrainImage, path_txtImageLabel, label_enc=label_enc,
    # data_aug=data_aug_ZSL,
)
# Make dataloader
dataloader_trn = torch.utils.data.DataLoader(
    dataset=dataset_trn, batch_size=params_batch_size_fWGAN,
    shuffle=True,
    # sampler=WeightedRandomSampler(np.ones(len(dataset_trn)), num_samples=len(dataset_trn)//100),
    num_workers = params_num_workers,pin_memory=True,
)


torch.cuda.set_device(params_cuda_device)
# Model & Loss
lst_tsClassEmb = [Tensor(emb).cuda() for emb in [arr_ClassNameVec, arr_ClassAttr]]
emb_size = [emb.shape[1] for emb in lst_tsClassEmb]
ConvNet = resnet50(num_classes=NClassTotal).cuda()
state_dict = torch.load(path_ptmdlConvNet,map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
state_dict = OrderedDict({
    key:state_dict[key]
    for key in state_dict
    if key in ConvNet.state_dict().keys()
})
ConvNet.load_state_dict(state_dict)

G = G_Feat(
    emb_size=emb_size,
    dimHid = params_dimHid_G,
    feat_size=params_dimVisFeat,
).cuda()
if params_LoadPretr_G:
    state_dict = torch.load(path_ptmdlWGAN_G,map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
    state_dict = OrderedDict({
        key:state_dict[key]
        for key in state_dict
        if key in G.state_dict().keys()
    })
    G.load_state_dict(state_dict)

D = D_EM(
    feat_size=params_dimVisFeat, emb_size=emb_size,
    dimHid = params_dimHid_G,
).cuda()
if params_LoadPretr_D:
    state_dict = torch.load(path_ptmdlWGAN_D,map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
    state_dict = OrderedDict({
        key:state_dict[key]
        for key in state_dict
        if key in D.state_dict().keys()
    })
    D.load_state_dict(state_dict)

criterion = loss_fWGAN(
    G, D, ConvNet,
    lambda_=params_fWGAN_lambda,
    beta=params_fWGAN_beta).cuda()
optimizer_G = torch.optim.Adam(
    filter(lambda p: p.requires_grad, G.parameters()),
    lr=params_lr_G, weight_decay=params_weight_decay_G,
)
optimizer_D = torch.optim.RMSprop(
    filter(lambda p: p.requires_grad, D.parameters()),
    lr=params_lr_D, weight_decay=params_weight_decay_D,
)

NEpochs = params_NEpochs_fWGAN
# NEpochs = 1
lrs_epoch = []
losses_epoch = []
valids_epoch = []

path_pklLog = os.path.join(path_folderExp, r'ExperimentLog.pkl')
path_fileModelStatePrev = ''
LastBest = False
path_fileModelState_G,path_fileModelState_G_best=None,None
path_fileModelState_D,path_fileModelState_D_best=None,None

print('******************* Training Start *******************')
with print_elapsed_time(prompt='Training part'):
    diff_valid_g_r_best = -np.inf
    for ith_epoch in range(NEpochs):
        # print('%d/%d'%(ith_epoch, NEpochs))
        time_start = time.time()
        stats_G = AverageMeter(NVars=2)
        stats_D = AverageMeter(NVars=4)

        ConvNet.eval()
        stats_G, stats_D,_,_= trainer_epoch_fWGAN(
            ConvNet, G, D,
            dataloader_trn, lst_tsClassEmb,
            criterion,
            optimizer_G=optimizer_G, optimizer_D=optimizer_D,
            training=True, NCritics=params_NCritics,
            stats_G=stats_G, stats_D=stats_D,
        )
        _,_,arr_feat_g_valid, arr_feat_r_valid= trainer_epoch_fWGAN(
            ConvNet, G, D,
            dataloader_trn, lst_tsClassEmb,
            criterion,
            training=False, NCritics=params_NCritics,
            stats_G=stats_G, stats_D=stats_D,
        )

        time_elapsed = time.time()-time_start

        loss_avg_G, loss_cls_avg = stats_G.avg()
        loss_avg_D, feat_g_valid_avg,feat_r_valid_avg, GP_avg = stats_D.avg()
        arr_diff_valid_g_r = arr_feat_g_valid-arr_feat_r_valid
        diff_valid_g_r = arr_diff_valid_g_r.mean()
        ratio_pos_diff_valid_g_r = (arr_diff_valid_g_r>0).sum()/arr_diff_valid_g_r.size

        EpochResult = r'lss_G_{:.3g}_D_{:.3g}_LssCls_{:.3g}_GP_{:.3g}_valid_g_{:.3g}_r_{:.3g}_diff_{:.3g}_PosRat_{:.3g}'.format(
            loss_avg_G, loss_avg_D, loss_cls_avg, GP_avg,
            arr_feat_g_valid.mean(), arr_feat_r_valid.mean(),diff_valid_g_r, ratio_pos_diff_valid_g_r)

        # print epoch log
        if ith_epoch % 1 == 0:
            print('Epoch {}/{}, {:.1f}[s] elapsed, {:s}'.format(
                ith_epoch+1, NEpochs, time_elapsed, EpochResult))
            print('Log saved at:', path_folderExp)
        losses_epoch.append((
            loss_avg_G, loss_avg_D,loss_cls_avg,GP_avg))
        valids_epoch.append((
            arr_feat_g_valid.mean(),arr_feat_r_valid.mean(), arr_diff_valid_g_r, ))
        # Save log
        dict_Log = {
            'lrs_epoch':lrs_epoch,
            'losses_epoch':losses_epoch,
            'valids_epoch':valids_epoch,
        }
        with open(path_pklLog, 'wb') as f:
            pickle.dump(dict_Log, f)
        # Save best & remove previous best
        UpdateBest = (np.abs(diff_valid_g_r) < np.abs(diff_valid_g_r_best)) \
                    and loss_cls_avg<2 \
                    and arr_feat_g_valid.mean()!=0 and arr_feat_g_valid.mean()!=0 \
                    and ratio_pos_diff_valid_g_r!=0 \
                    and (ith_epoch>=params_Epoch_SaveBestEnable)
        if UpdateBest:
            diff_valid_g_r_best=diff_valid_g_r
            if path_fileModelState_G_best:
                os.remove(path_fileModelState_G_best)
            if path_fileModelState_D_best:
                os.remove(path_fileModelState_D_best)
        if path_fileModelState_G!=path_fileModelState_G_best:
            os.remove(path_fileModelState_G)
        if path_fileModelState_D!=path_fileModelState_D_best:
            os.remove(path_fileModelState_D)
        # Save current epoch model
        model_result = r'Ep_{:03d}_{:s}'.format(
            ith_epoch+1,  EpochResult,)
        path_fileModelState_G = os.path.join(
            path_folderExpBestModel, r'{:s}_G.ptmdl'.format(model_result, ),)
        path_fileModelState_D = os.path.join(
            path_folderExpBestModel, r'{:s}_D.ptmdl'.format(model_result, ),)
        if UpdateBest:
            path_fileModelState_G_best=path_fileModelState_G
            path_fileModelState_D_best=path_fileModelState_D
        torch.save(G.state_dict(), path_fileModelState_G)
        torch.save(D.state_dict(), path_fileModelState_D)
