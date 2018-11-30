# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：path configuration
# Author: Yinda XU

import os
from .config_local import path_folderDataset, path_ptmdlConvNet

path_folderSrc = os.path.dirname(os.path.dirname(__file__))
path_folderPrj = os.path.dirname(path_folderSrc)

# if not os.path.exists(path_folderDataset):
#     raise RuntimeError('{} not found, \n\
#     please set the path_folderDataset in ./config/path.py to the real dataset location'.format(path_folderDataset))

path_txtAttrList = os.path.join(path_folderDataset, r'attribute_list.txt')
path_txtAttrPerClass = os.path.join(path_folderDataset, r'attributes_per_class.txt')
path_txtClassNameVec = os.path.join(path_folderDataset, r'class_wordembeddings_300d.txt')
path_txtLabelList = os.path.join(path_folderDataset, r'label_list.txt')
path_txtImageLabel = os.path.join(path_folderDataset, r'train.txt')
path_txtImage = os.path.join(path_folderDataset, r'image.txt')
path_txtImageLabelCleaned = os.path.join(path_folderDataset, r'train_cleaned.txt')

path_csvClassAttrEnc = os.path.join(path_folderDataset, r'ClassAttrEnc.csv')

path_folderTrainImage = os.path.join(path_folderDataset, r'train')
path_folderTestImage = os.path.join(path_folderDataset, r'test')

# split
# path_folderDatasetConfig = os.path.join(path_folderPrj, r'DataConfig')
path_folderDatasetConfig = os.path.join(path_folderDataset, r'DataConfig')
if not os.path.exists(path_folderDatasetConfig):
    os.makedirs(path_folderDatasetConfig)
    print(path_folderDatasetConfig, 'created.')
path_npyIndexSeenTrn = os.path.join(path_folderDatasetConfig, r'index_SeenTrn.npy')
path_npyIndexSeenVal = os.path.join(path_folderDatasetConfig, r'index_SeenVal.npy')
path_npyIndexUnseenTst = os.path.join(path_folderDatasetConfig, r'index_UnseenTst.npy')
path_npyCatSeen = os.path.join(path_folderDatasetConfig, r'arr_CatSeen.npy')
path_npyCatUnseen = os.path.join(path_folderDatasetConfig, r'arr_CatUnseen.npy')
path_npyCatAnnotd = os.path.join(path_folderDatasetConfig, r'arr_CatAnnotd.npy')

path_csvLabelEnc = os.path.join(path_folderDatasetConfig, r'LabelEnc.csv')
path_csvLabelEncInv = os.path.join(path_folderDatasetConfig, r'LabelEncInv.csv')

path_txtCoarseClassList = os.path.join(path_folderDataset, r'CoarseClassList.txt')

# result
# parent folder
path_folderResult = os.path.join(path_folderPrj, r'Result')
if not os.path.exists(path_folderResult):
    os.makedirs(path_folderResult)
    print(path_folderResult, 'created.')

# ConvNet weight
