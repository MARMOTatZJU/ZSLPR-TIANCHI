# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：optimizer configuration
# Author: Yinda XU

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
fontdict_axis = {'fontsize':24}
fontdict_title = {'fontsize':32}
plt.rcParams["figure.dpi"] = 300

import torch
import torch.optim
params_SaveBest = True
params_Epoch_SaveBestEnable = 5

params_useLRScheduler = True
params_lr = 7e-2
params_weight_decay = 5e-3
params_func_lr = lambda n : (1-n/params_EpochInflect*(1-params_lrInflect**1.1))**0.9 \
    if n<params_EpochInflect \
    else params_lrInflect*0.3**(1+(n-params_EpochInflect)//2)
    # else params_lrInflect*(1-(n-params_EpochInflect)/(params_NEpochs-params_EpochInflect))
params_lrInflect = 0.01
params_NEpochs = 200
params_NEpochs_TriHardBatch = params_NEpochs//2
params_NEpochs_Inflect = 6
params_EpochInflect=params_NEpochs-params_NEpochs_Inflect
params_EpochTriHardBatch = params_NEpochs-params_NEpochs_TriHardBatch

params_batch_size = 32
params_batch_size_TriHard = 64
params_batch_size_fWGAN = 32
params_numDataAug = 512

# WGAN-GP configuration
params_lr_G = 1e-5
params_weight_decay_G = 2e-4
params_lr_D = 1e-5
params_weight_decay_D = 2e-4
params_NEpochs_fWGAN = 300
params_NCritics = 5
