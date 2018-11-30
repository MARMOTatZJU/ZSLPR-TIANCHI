# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：function for backing up
# Author: Yinda XU

import os
import shutil

from config.path import *
import utils
from utils.wheels import *

def createBackUp(title='Exp'):
    '''
    Make the experiment result folder & backup config
    Parameters
        title: string
            Experiment title
    Returns
        TimeStamp: string
            Time stamp of the current at moment of creation
        path_folderExp: string
            Path of experiment result folder
        path_folderExpBestModel: string
            Path of best model folder
    '''
    TimeStamp = getTimeStamp()
    name_FolderExp = '{}_{}'.format(title, TimeStamp)
    # Create the experiment folder
    path_folderExp = os.path.join(path_folderResult, name_FolderExp,)
    if os.path.exists(path_folderExp):
        shutil.rmtree(path_folderExp)
    os.makedirs(path_folderExp)
    print('Experiment result to be saved at:', path_folderExp)
    # Create best model folder of the experiment
    path_folderExpBestModel = os.path.join(path_folderExp, 'BestModel')
    os.makedirs(path_folderExpBestModel)

    # Create the src folder of the experiment
    path_folderExpSrc = os.path.join(path_folderExp, 'Src')
    # Backup all the config files
    path_folderCpSrc = path_folderSrc
    path_folderCpDst = path_folderExpSrc
    shutil.copytree(path_folderCpSrc, path_folderCpDst)

    return TimeStamp, path_folderExp, path_folderExpBestModel
