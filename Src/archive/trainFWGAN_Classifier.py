# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：run training of fWGAN's classifier
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
from torch.utils.data import Dataset, DataLoader

# import in-house packages
from dataset import datasetZSL, datasetZSL_wtDataAug, datasetFWGAN_FeatGen
import models
from models import resnet50, G_Feat, D_EM, fWGAN_classifier

# import config
import config
from config import *

import utils
from utils.wheels import *
from utils.trainer import trainer_epoch
from utils.DataAug import data_aug_ZSL
# Make experiment result folder
TimeStamp, path_folderExp, path_folderExpBestModel = utils.backup.createBackUp()

from prepareData import *

# Make dataset
dataset_trn = datasetZSL(
    path_folderTrainImage, path_txtImageLabel, label_enc=label_enc,
    # data_aug=data_aug_ZSL,
)
lst_tsClassEmb = [Tensor(emb).cuda() for emb in [arr_ClassNameVec, arr_ClassAttr]]
emb_size = [emb.shape[1] for emb in lst_tsClassEmb]

G = G_Feat(
    emb_size=emb_size,
    dimHid = params_dimHid_G,
    feat_size=params_dimVisFeat,
).cuda()

state_dict = torch.load(path_ptmdlWGAN_G,map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
state_dict = OrderedDict({
    key:state_dict[key]
    for key in state_dict
    if key in G.state_dict().keys()
})
G.load_state_dict(state_dict)
G=G.cuda()
dataset_trn_FeatGen=datasetFWGAN_FeatGen(lst_tsClassEmb, G, idxCatTotal, NInstancesPerCls=int(NInstancesPerCls_mean/5))
dataset_prd = datasetZSL(path_folderTestImage, path_txtImage, index_split=index_UnseenPrd, label_enc=label_enc, DummyTarget=True)

# Make dataloader
dataloader_trn = torch.utils.data.DataLoader(
    dataset=dataset_trn, batch_size=64, shuffle=True,
    num_workers = params_num_workers,pin_memory=True,
)
dataloader_trn_FeatGen = torch.utils.data.DataLoader(
    dataset=dataset_trn_FeatGen, batch_size=64, shuffle=True,
    num_workers = params_num_workers,
    # pin_memory=True,
)
dataloader_prd = torch.utils.data.DataLoader(
    dataset=dataset_prd, batch_size=params_batch_size, shuffle=False,
    num_workers = params_num_workers,pin_memory=True,
)


torch.cuda.set_device(params_cuda_device)
# Model & Loss
# ts_ClassEmbTotal = Tensor(arr_ClassNameVec).cuda()
ConvNet = resnet50(num_classes=NClassTotal).cuda()
state_dict = torch.load(path_ptmdlConvNet,map_location={'cuda:1' : 'cuda:0', 'cuda:2' : 'cuda:0'})
state_dict = OrderedDict({
    key:state_dict[key]
    for key in state_dict
    if key in ConvNet.state_dict().keys()
})
# ConvNet.load_state_dict(state_dict)

classifier = fWGAN_classifier(params_dimVisFeat, NClassTotal).cuda()

criterion = nn.CrossEntropyLoss()
lrs_epoch = []
losses_epoch = []
accus_epoch = []

optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, classifier.parameters()),
    lr=1e-2, weight_decay=1e-3,
)
NEpochs = 6
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda n:(1-n/NEpochs)**0.9
)
lrs_epoch = []
losses_epoch = []
accus_epoch = []

path_pklLog = os.path.join(path_folderExp, r'ExperimentLog.pkl')

print('******************* Training Start *******************')
with print_elapsed_time(prompt='Training part'):
    accu_best = 0
    if True:
        loss_avg_val, accu_avg_val = trainer_epoch(
            classifier, dataloader_trn_FeatGen, criterion,
        )
    print('Init cond:', loss_avg_val, accu_avg_val)

    for ith_epoch in range(NEpochs):
        time_start = time.time()
        loss_avg_trn, loss_avg_val, loss_avg_tst, accu_avg_val, accu_avg_tst = \
            0,0,0,0,0
        if params_useLRScheduler:
            lr_scheduler.step()
        loss_avg_trn, accu_avg_trn, loss_avg_val, accu_avg_val = 0,0,0,0
        loss_avg_tst, accu_avg_tst, loss_avg_prd, accu_avg_prd = 0,0,0,0
        arr_outputs_prd = None

        with torch.cuda.device(params_cuda_device):
# train model for one epoch
            if True:
                loss_avg_trn, accu_avg_trn = trainer_epoch(
                    classifier, dataloader_trn_FeatGen, criterion,
                    training=True, optimizer=optimizer,
                )
# # validate model with validation set
            if True:
                loss_avg_val, accu_avg_val = trainer_epoch(
                    classifier, dataloader_trn_FeatGen, criterion,
                )
# # Test
            if False:
                loss_avg_tst, accu_avg_tst = trainer_epoch(
                    model, dataloader_tst, criterion,
                    idxCat=LongTensor(idxCatUnseen).cuda(),
                )
# Predict
            if True and not params_LocalTest:
                state_dict['fc.weight'] = classifier.state_dict()['fc.weight']
                state_dict['fc.bias'] = classifier.state_dict()['fc.bias']
                ConvNet.load_state_dict(state_dict)
                arr_outputs_prd = trainer_epoch(
                    ConvNet,
                    dataloader_prd,
                    criterion,
                    predict=True,
                    predict_DataAug=False,
                    idxCat=LongTensor(idxCatUnannotd).cuda(),
                )

        time_elapsed = time.time()-time_start

        EpochResult = r'TRN_lss_{:.3g}_accu_{:.3g}_VAL_lss_{:.3g}_accu_{:.3g}_TST_lss_{:.3g}_accu_{:.3g}'.format(
            loss_avg_trn,accu_avg_trn, loss_avg_val, accu_avg_val, loss_avg_tst, accu_avg_tst,)

        # print epoch log
        lr_epoch = optimizer.param_groups[0]['lr']
        if ith_epoch % 1 == 0:
            print('Epoch {}/{}, {:.1f}[s] elapsed, {:s}, lr: {:.3g}'.format(
                ith_epoch+1, NEpochs, time_elapsed, EpochResult, lr_epoch,))
            print('Log saved at:', path_folderExp)
        lrs_epoch.append(lr_epoch)
        losses_epoch.append((loss_avg_trn, loss_avg_val, loss_avg_tst))
        accus_epoch.append((accu_avg_trn, accu_avg_val, accu_avg_tst))
        # Save log
        dict_Log = {
            'lrs_epoch':lrs_epoch,
            'losses_epoch':losses_epoch,
            'accus_epoch':accus_epoch,
        }
        with open(path_pklLog, 'wb') as f:
            pickle.dump(dict_Log, f)

        # Save best
        model_result = r'{:s}_Ep_{:d}_{:s}'.format(
            type(classifier).__name__, ith_epoch+1,  EpochResult
        )
        path_fileModelState = os.path.join(
            path_folderExpBestModel, r'{:s}.ptmdl'.format(model_result, ),
        )
        torch.save(classifier.state_dict(), path_fileModelState)
        accu_best = accu_avg_val
        # Save predict result
        if not isinstance(arr_outputs_prd, type(None)):
            sr_outputs_prd = pd.Series(sr_LabelEncInv[arr_outputs_prd.squeeze()].values, index=sr_Image[index_UnseenPrd].index)
            path_txtSub = os.path.join(
                path_folderExp, r'submit_{:s}.txt'.format(model_result)
            )
            sr_outputs_prd.to_csv(path_txtSub, sep='\t')
''
