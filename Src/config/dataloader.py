# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：dataloader configuration
# Author: Yinda XU

import platform
import os

if 'windows' in platform.platform().lower():
    params_num_workers = 0 # must be 0 under Windows
    print('num_workers set to zero for Windows detected, ')
else:
    params_num_workers = int(os.cpu_count()*0.75) # equal to cpu processor number to accelerate
    print('num_workers set to {:d}'.format(params_num_workers))
