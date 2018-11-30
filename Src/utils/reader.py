# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：read annotation files(eg. class name embedding, class attribute)
#                and transform into numpy array
# Author: Yinda XU

# for reading, use the following code:
    # path_txtAttrList = os.path.join(path_folderDataset, r'attribute_list.txt')
    # path_txtAttrPerClass = os.path.join(path_folderDataset, r'attributes_per_class.txt')
    # path_txtClassNameVec = os.path.join(path_folderDataset, r'class_wordembeddings.txt')
    # path_txtLabelList = os.path.join(path_folderDataset, r'label_list.txt')
#     sr_LabelList = utils.reader.readLabelList(path_txtLabelList)
#     sr_ImageLabel = utils.reader.readImageLabel(path_txtImageLabel)
#     sr_ClassAttr = utils.reader.readClassAttr(path_txtAttrPerClass)
#     sr_ClassNameVec = utils.reader.readClassAttr(path_txtClassNameVec)


import numpy as np
import pandas as pd

def readAttrList(path):
    '''
    Read 'attribute_list.txt'
    Parameters
        path: string
            path of .txt file
    Returns
        sr_LabelList: Series of length=C
            Attribute ID in index
            Attribute name in values
    '''
    df_AttrList = pd.read_csv(path, sep='\t', header=None, index_col=0, names=['AttributeID', 'AttributeName', ])
    sr_AttrList = df_AttrList['AttributeName']
    return sr_AttrList

def readClassAttr(path, arr_AttrList=None, arr_LabelList=None, VecAsList=True):
    '''
    Read 'attributes_per_class.txt'
    Parameters
        path: string
            path of .txt file
    Returns
        sr_ClassAttr: Series of length=C
            class attributes (C categories in index, attribute vector as list in values)
    '''
    df_ClassAttr = pd.read_csv(path, sep='\t', header=None, index_col=0, )
    df_ClassAttr.index.name='Class'
    # align with the label list provided
    if arr_LabelList:
        assert set(df_ClassAttr.index) == set(arr_LabelList)
        df_ClassAttr = df_ClassAttr.reindex(arr_LabelList)
    # sort label  in case of no label list provided
    else:
        df_ClassAttr.sort_index(inplace=True)
    if not isinstance(arr_AttrList, type(None)):
        df_ClassAttr.columns = arr_AttrList
    if VecAsList:
        df_ClassAttr['ClassAttr'] = df_ClassAttr.values.tolist()
        sr_ClassAttr = df_ClassAttr['ClassAttr']
        sr_ClassAttr.index.name='Class'
        return sr_ClassAttr
    else:
        return df_ClassAttr


def readClassNameVec(path, sr_LabelList, VecAsList=True):
    '''
    Read 'class_wordembeddings.txt'
    Parameters
        path: string
            path of .txt file
        arr_LabelList: numpy array of shape=(C, )
            Label (C categories)
        arr_LabelName: znumpy array of shape=(C, )
            Label name (C categories)
    Returns
        sr_ClassNameVec: Series of length=C
            class name word embeddings
            (C categories in index, class name word embeddings as list in values)
    '''
    df_ClassNameVec = pd.read_csv(path, sep='\t', header=None, index_col=0)
    try:
        assert set(df_ClassNameVec.index) == set(sr_LabelList.values) # align with the label list provided
    except AssertionError:
        df_ClassNameVec = pd.read_csv(path, sep=' ', header=None, index_col=0)
    assert set(df_ClassNameVec.index) == set(sr_LabelList.values) # align with the label list provided
    df_ClassNameVec['Label'] = pd.Series(sr_LabelList.index.values, index=sr_LabelList.values)
    df_ClassNameVec.index=df_ClassNameVec['Label']
    del df_ClassNameVec['Label']
    df_ClassNameVec = df_ClassNameVec.reindex(sr_LabelList.index )
    if VecAsList:
        df_ClassNameVec['ClassNameVec'] = df_ClassNameVec.values.tolist()
        sr_ClassNameVec = df_ClassNameVec['ClassNameVec']
        sr_ClassNameVec.index.name='ClassNameVec'
        return sr_ClassNameVec
    else:
        return df_ClassNameVec

def readLabelList(path):
    '''
    Read 'label_list.txt'
    Parameters
        path: string
            path of .txt file
    Returns
        sr_LabelList: Series of length=C
            Label in index
            Label name in values
    '''
    df_LabelList = pd.read_csv(path, sep='\t', header=None, index_col=0, names=['Class', 'LabelName', ])
    sr_LabelList = df_LabelList['LabelName']
    return sr_LabelList

def readImageLabel(path, DummyTarget=False, ):
    '''
    Read 'train.txt'
    Parameters
        path: string
            path of .txt file
    Returns
        sr_ImageLabel: Series of length=N
            Image file name in index
            Image label in values
    '''
    df_ImageLabel = pd.read_csv(path, sep='\t', header=None, index_col=0, names=['Filename', 'Label', ])
    if DummyTarget:
        df_ImageLabel['Label']=-1
    sr_ImageLabel = df_ImageLabel['Label']
    return sr_ImageLabel
