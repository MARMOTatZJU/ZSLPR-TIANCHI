# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning competition
# Team: AILAB-ZJU
# Code function：encoding embeddings
# Author: Yinda XU

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

CatAttrFreqThres = 5

def attribute_enc(df_ClassAttr, sr_AttrList):
    sr_ClassAttrFreq = df_ClassAttr.nunique(axis=0)
    oneHotEncoder = OneHotEncoder(sparse=False)
    labelEncoder = LabelEncoder()
    NColsCat = 0
    lst_ClassAttrEnc = []
    lst_colEnc = []
    for col, val in sr_ClassAttrFreq.items():
        AttrName = sr_AttrList[col]
        if val<=CatAttrFreqThres:
            NColsCat+=1
            EmbEncoded = oneHotEncoder.fit_transform(
                labelEncoder.fit_transform(
                    df_ClassAttr[[col]]).reshape(-1, 1),)
            colsEnc = ['%s.%d'%(AttrName, ith) for ith in range(val)]
        else:
            EmbEncoded = df_ClassAttr[[col]].values
            colsEnc = [AttrName]
        lst_ClassAttrEnc.append(EmbEncoded)
        lst_colEnc.extend(colsEnc)
    arr_ClassAttrEnc = np.concatenate(lst_ClassAttrEnc, axis=1)
    df_ClassAttrEnc = pd.DataFrame(arr_ClassAttrEnc, index=df_ClassAttr.index, columns=lst_colEnc)
    return df_ClassAttrEnc
