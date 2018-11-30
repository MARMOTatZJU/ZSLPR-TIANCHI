# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：model configuration
# Author: Yinda XU

params_dimVisFeat = 2048

params_useTriHardLoss = True
params_TriHardMargin = 0.3
params_coeff_TriHardLoss = 5

params_useTestAug = False

params_fWGAN_beta = 0.05
params_fWGAN_lambda = 10

params_dimHid_G = [1024, 4096]
params_dimHid_D = [512, 128]

params_LoadPretr_G = False
params_LoadPretr_D = False
