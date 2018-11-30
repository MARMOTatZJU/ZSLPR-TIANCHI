# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：dataset & dataloader
# Author: Yinda XU

import os
import cv2

import numpy as np

import torch
from torch import Tensor
from torch import LongTensor
from torch.utils.data import Dataset

import utils


def loaderDftZSL(image, label):
    '''
    Default loader for datasetZSL
        including normalization for 8-bit image & tensor transformation
    Parameters
        image: numpy array of shape=(H, W, C)
            Image after data augmentation
        label: numpy array of size=1
            Label
    Returns
        ts_image: tensor of shape=(C, H, W)
            Tensor of image ready to be output
        ts_label: tensor of shape=(1, )
            Tensor of label ready to be output
    '''
    if len(image.shape)==1:
        image = np.asarray([image, image, image])
    else:
        image = np.moveaxis(image, 2, 0)
    image = image/255 # minmax normalization
    image = (image-image.mean())/image.std() # standardization
    ts_image, ts_label=Tensor(image), LongTensor(label.reshape(-1))

    return ts_image, ts_label


class datasetZSL(Dataset):
    '''
    Make Pytorch dataset for ZSL images
    Implementation unfinished
    '''
    def __init__(self, path_folderImage, path_txtImageLabel, DummyTarget=False, index_split=None, data_aug=None, label_enc=None, loader=loaderDftZSL):
        '''
        Constructor of datasetZSL
        Parameters
            arr_Image,
            arr_Label,
            index_split: pandas index of split
                Make
            data_aug: function of format: image=func(image)
                data augmentation
            label_enc: function of format: label=func(label)
                label encoder
            loader: function of format: ts_image, ts_label=func(image, label)
                transform into tensors
        '''
        self.path_folderImage = path_folderImage
        sr_ImageLabel=utils.reader.readImageLabel(path_txtImageLabel, DummyTarget=DummyTarget)
        if not isinstance(index_split, type(None)):
            sr_ImageLabel = sr_ImageLabel[index_split]
        self.arr_ImageName = sr_ImageLabel.index.values
        self.arr_ImageLabel = sr_ImageLabel.values
        self.data_aug = data_aug
        if DummyTarget:
            self.label_enc = None
        else:
            self.label_enc = label_enc
        self.loader = loader
    def __getitem__(self, idx):
        '''
        Dataset instance indexing method
        Parameters
            idx: instance index
        Returns
            ts_image: tensor
                Tensor of image
            ts_label: tensor
                Tensor of label
        '''
        path_Image = os.path.join(self.path_folderImage, self.arr_ImageName[idx])
        image = cv2.imread(path_Image, cv2.IMREAD_COLOR)
        label = np.asarray(self.arr_ImageLabel[idx]).reshape(-1, )
        # data augmentation
        if self.data_aug:
            image=self.data_aug([image])[0]
        # encode label
        if self.label_enc:
            label=self.label_enc(label)
        try:
            ts_image, ts_label = self.loader(image, label)
        except:
            print(path_Image, 'loading failed')
        return ts_image, ts_label
    def __len__(self, ):
        '''
        Get dataset length
        Returns
            length: integer
        '''
        return self.arr_ImageLabel.size

class datasetZSL_wtDataAug(Dataset):
    '''
    Wrap the datasetZSL into test data augmentation version:
    '''
    def __init__(self, dataset, num_DataAug = 16):
        self.dataset=dataset
        self.num_DataAug = num_DataAug
    def __getitem__(self, idx):
        return self.dataset[idx//self.num_DataAug]
    def __len__(self, ):
        return len(self.dataset)*self.num_DataAug

class datasetFWGAN_FeatGen(Dataset):
    '''
    Generate visual features given a class embedding
    '''
    def __init__(self,
        lst_ClassEmb, G_fWGAN,
        idxClasses, NInstancesPerCls=200,
    ):
        self.lst_ClassEmb = lst_ClassEmb
        self.G=G_fWGAN
        self.NInstancesPerCls=NInstancesPerCls
        self.idxClasses=idxClasses
    def __getitem__(self, idx):
        idx//=self.NInstancesPerCls
        ClassEmb = [ emb[[idx], :] for emb in self.lst_ClassEmb]

        FeatGen = self.G(ClassEmb).squeeze(dim=0)
        target = LongTensor([self.idxClasses[idx]])
        return FeatGen, target
    def __len__(self, ):
        return self.NInstancesPerCls*self.idxClasses.size
