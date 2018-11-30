# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：dataset making related functions
# Author: Yinda XU

import os
import numpy as np
import pandas as pd
import cv2

import sklearn
from sklearn.model_selection import train_test_split

import utils
from utils.reader import readImageLabel

unseen_ratio_dft = 0
valid_ratioa_dft = 0.01

# Deprecated
def stackImages(path_folderImage, path_txtImageLabel, ):
    '''
    !!!Deprecated!!!
    Read image files & merge
    Parameters
        path_folderImage: string
        path_txtImageLabel: string
    Returns
        arr_ImageTotal: numpy array of shape=(N, H, W, C)
            stacked image array of OpenCV style H*W*C for N instances
        arr_ImageLabel: numpy array of shape=(N, )
            image label
    '''
    arr_ImageName, arr_ImageLabel = readImageLabel(path_txtImageLabel)
    assert arr_ImageName.size == arr_ImageLabel.size
    lst_arr_img = []
    num_img = arr_ImageName.size
    freq_print = num_img//100
    print('{} images to be stacked.'.format(num_img))
    for ith, filename in enumerate(arr_ImageName):
        if ith % freq_print == 0:
            print('{}/{}'.format(ith, num_img))
        arr_img = cv2.imread(os.path.join(path_folderImage, filename))
        lst_arr_img.append(arr_img)
    arr_ImageTotal = np.stack(lst_arr_img, axis=0)
    path_folderSave = os.path.dirname(path_txtImageLabel)
    return arr_ImageTotal, arr_ImageLabel

def makeMergedTabel(path_txtLabelList, path_txtAttrPerClass, path_txtClassNameVec, path_txtImageLabel):
    '''
    Make merged table
    Parameters
        path_txtLabelList: path of 'label_list.txt'
        path_txtAttrPerClass: path of 'attributes_per_class.txt'
        path_txtClassNameVec: path of 'class_wordembeddings.txt'
        path_txtImageLabel: path of 'train.txt'
    Returns
        df_Dataset: DataFrame of shape=(N, 3)
            Index
                'Filename': filenames of images
            Columns
                'ClassAttr': class attributes
                'ClassNameVec': class name word embeddings
                'ImageLabel': image labels
    '''
    sr_LabelList = utils.reader.readLabelList(path_txtLabelList)
    sr_ClassAttr = utils.reader.readClassAttr(path_txtAttrPerClass)
    sr_ClassNameVec = utils.reader.readClassNameVec(path_txtClassNameVec, sr_LabelList.index.values, sr_LabelList.values)
    sr_ImageLabel = utils.reader.readImageLabel(path_txtImageLabel)
    df_MergedTable = pd.DataFrame(
        {
            'ImageLabel':sr_ImageLabel.values,
            'ClassAttr':sr_ClassNameVec[sr_ImageLabel].values,
            'ClassNameVec':sr_ClassNameVec[sr_ImageLabel].values,
        },
        index=sr_ImageLabel.index,
    ).sort_index()
    return df_MergedTable

def cat_split(arr_ImageLabel, unseen_ratio=unseen_ratio_dft):
    '''
    Split categories into 'seen' and 'unseen'
    Parameters
        arr_ImageLabel: numpy array of size=N,
            Label list to be splitted which has C unique elements (categories)
    Returns
        arr_CatSeen: numpy  array of shape (C-M, )
            Label list to be seen, M = [C*unseen_ratio]
        arr_CatUnseen: numpy  array of shape (M, )
            Label list to be unseen
    '''
    arr_ImageCat = np.unique(arr_ImageLabel.reshape(-1))
    np.random.shuffle(arr_ImageCat)
    idx_split = int(len(arr_ImageCat)*(1-unseen_ratio))
    arr_CatSeen, arr_CatUnseen = arr_ImageCat[:idx_split], arr_ImageCat[idx_split:]
    return arr_CatSeen, arr_CatUnseen

def dataset_split(sr_ImageLabel, arr_CatSeen, arr_CatUnseen):
    '''
    Split dataset w.r.t seen categories & unseen categories given
    Parameters
        sr_ImageLabel: Series of shape=(N,)
            Column 'ImageLabel' of the previously made dataset
        arr_CatSeen: numpy array of shape (Cseen, )
            List of seen categories
        arr_CatUnseen: numpy array of shape (Cunseen, )
            List of unseen categories
    Returns
        index_Seen: pandas index of length=Nseen
            Indices of seen set
        index_Unseen: pandas index of length=Nunseen
            Indices of unseen set
    '''
    index_Seen = sr_ImageLabel[sr_ImageLabel.isin(arr_CatSeen)].index
    index_Unseen = sr_ImageLabel[sr_ImageLabel.isin(arr_CatUnseen)].index
    return index_Seen, index_Unseen

def makeSplit(sr_ImageLabel, path_folderSaveSplit=None, unseen_ratio=unseen_ratio_dft, valid_ratio=valid_ratioa_dft):
    '''
    Make a complete split for ZSL & save the split
    Parameters
        sr_ImageLabel: Series of shape=(N,)
            Column 'ImageLabel' of the previously made dataset
        path_folderSaveSplit：string
            path of folder where to save split
        unseen_ratio: float in [0, 1]
            Ratio of unseen categories
        valid_ratio: float in [0, 1]
            Ratio of samples reserved for validation
    Returns
        index_SeenTrn: pandas index of length=Nseen_Trn
            Indices of training set in seen set
        index_SeenVal: pandas index of length=Nunseen_Val
            Indices of validation set in seen set
        index_UnseenTst: pandas index of length=Nunseen
            Indices of unseen set
        arr_CatSeen: numpy array of shape=(C-M, )
            Seen categories
        arr_CatUnseen: numpy array of shape=(M, )
            Unseen categories
    '''
    arr_CatSeen, arr_CatUnseen = utils.dataset.cat_split(sr_ImageLabel.values, unseen_ratio=unseen_ratio)
    index_Seen, index_UnseenTst = utils.dataset.dataset_split(sr_ImageLabel, arr_CatSeen, arr_CatUnseen)
    index_SeenTrn, index_SeenVal = train_test_split(index_Seen, test_size=valid_ratio, stratify=sr_ImageLabel[index_Seen])
    if path_folderSaveSplit:
        dict_save = {
            'index_SeenTrn': index_SeenTrn,
            'index_SeenVal': index_SeenVal,
            'index_UnseenTst': index_UnseenTst,
            'arr_CatSeen':arr_CatSeen,
            'arr_CatUnseen':arr_CatUnseen,
        }
        for key, val in dict_save.items():
            path_save = os.path.join(path_folderSaveSplit, r'{:s}.npy'.format(key))
            if isinstance(val, (pd.Index, pd.Series, pd.DataFrame)):
                np.save(path_save, val.values)
            else:
                np.save(path_save, val)
            print(path_save, 'saved.')
    return index_SeenTrn, index_SeenVal, index_UnseenTst, arr_CatSeen, arr_CatUnseen,
