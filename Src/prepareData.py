import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# import config
import config
from config import *

import utils
from utils.wheels import *
from utils.DataAug import data_aug_ZSL
# from utils.encoding import attribute_enc

sr_LabelList = utils.reader.readLabelList(path_txtLabelList)
# sr_ImageLabel = utils.reader.readImageLabel(path_txtImageLabel)
if os.path.exists(path_txtImageLabelCleaned):
    sr_ImageLabel = utils.reader.readImageLabel(path_txtImageLabelCleaned)
    print('Cleaned dataset loaded.')
else:
    sr_ImageLabel = utils.reader.readImageLabel(path_txtImageLabel)
NInstancesPerCls_mean = int(sr_ImageLabel.value_counts().values.mean())
sr_Image = utils.reader.readImageLabel(path_txtImage, DummyTarget=True)
df_ClassNameVec = utils.reader.readClassNameVec(path_txtClassNameVec, sr_LabelList, VecAsList=False)
_, sizeEmbed = df_ClassNameVec.shape
print('Class name embedding size: %d '%sizeEmbed)

# if os.path.exists(path_txtAttrList):
#     sr_AttrList = utils.reader.readAttrList(path_txtAttrList)
# if os.path.exists(path_txtAttrPerClass):
#     df_ClassAttr = utils.reader.readClassAttr(path_txtAttrPerClass, VecAsList=False)

if os.path.exists(path_csvLabelEnc):
    sr_LabelEnc = pd.read_csv(path_csvLabelEnc, index_col=0, header=None,squeeze=True)
    sr_LabelEncInv = pd.Series(sr_LabelEnc.index, index=sr_LabelEnc.values)

label_enc = lambda x:sr_LabelEnc.get(x).values
if os.path.exists(path_txtCoarseClassList):
    sr_LabelEnc_CoarseClasses = pd.read_csv(path_txtCoarseClassList, '\t', index_col=0, header=None, squeeze=True)

arr_CatTotal = sr_LabelEnc.index.values
NClassTotal = arr_CatTotal.size
print('Total {} classes'.format(NClassTotal))

# arr_CatSeen=np.load(path_npyCatUnseen)
# arr_CatUnseen = np.load(path_npyCatUnseen)
arr_CatAnnotd = np.load(path_npyCatAnnotd)
arr_CatUnannotd = np.asarray(list(set(arr_CatTotal)-set(arr_CatAnnotd)))
print('{} classes annoted'.format(arr_CatAnnotd.size))
print('{} classes unannoted'.format(arr_CatUnannotd.size))

# idxUnseen = sr_LabelEnc[arr_CatUnseen].values.astype(np.int) # for local test
# idxCatSeen = sr_LabelEnc[arr_CatSeen].values.astype(np.int) # for submission
# idxCatUnseen = sr_LabelEnc[arr_CatUnseen].values.astype(np.int) # for submission
idxCatAnnotd = np.arange(arr_CatAnnotd.size)
idxCatUnannotd = sr_LabelEnc[arr_CatUnannotd].values.astype(np.int) # for submission
idxCatTotal = np.arange(NClassTotal)

# align embeddings array
df_ClassNameVec = df_ClassNameVec.loc[arr_CatTotal, :] # alignment
# arr_ClassNameVec = np.asarray([elem for elem in sr_ClassNameVec.values])
arr_ClassNameVec = df_ClassNameVec.values
# Embed class attributes
# df_ClassAttr = df_ClassAttr.loc[arr_CatTotal, :] # alignment
# df_ClassAttrEnc = attribute_enc(df_ClassAttr, sr_AttrList)
# df_ClassAttrEnc = pd.read_csv(path_csvClassAttrEnc, index_col=0)
# arr_ClassAttr = df_ClassAttrEnc.values
# arr_ClassAttr = df_ClassAttr.values
# print('Number of categorical attributes: %d'%NColsCat)
# print('Attributes embedding size: %d'%arr_ClassAttr.shape[1])

# Class embeddings
# arr_ClassEmb = np.concatenate([arr_ClassNameVec, arr_ClassAttr], axis=1)
arr_ClassEmb = arr_ClassNameVec

assert (df_ClassNameVec.index == pd.Index(arr_CatTotal)).all()
# assert (df_ClassAttr.index == pd.Index(arr_CatTotal)).all()
# class number


# Load Dataset Split
if params_LocalTest: # local test
    index_SeenTrn = np.load(path_npyIndexSeenTrn)
    index_SeenVal = np.load(path_npyIndexSeenVal)
    index_UnseenTst = np.load(path_npyIndexUnseenTst)
    print('Local test, load prepared Seen/Unseen data split.')
else:
    index_UnseenTst = []
    if params_SelfValid: # submission
        index_SeenTrn, index_SeenVal = train_test_split(
            sr_ImageLabel.index.values, test_size=0.05, random_state=17, stratify=sr_ImageLabel.values,
        )
        print('No local test, reserve a small part for valid.')
    else:
        index_SeenTrn = sr_ImageLabel.index.values
        index_SeenVal = np.asarray([])
        print('No local test, neither validation.')

index_UnseenPrd = sr_Image.index.values
