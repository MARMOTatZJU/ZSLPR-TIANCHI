import numpy as np

import torch
from torch import Tensor
from torch import LongTensor
from torch.nn import functional as F

from .wheels import AverageMeter

def trainer_batch(
    model, data_batch, criterion,
    training, optimizer,
    predict, predict_DataAug,
    idxCat,
):
    inputs, targetsIdx=data_batch
    batch_size = inputs.shape[0]
    if not predict:
        targetsIdx = targetsIdx.cuda()
    if training:
        optimizer.zero_grad()

    outputs = model(inputs.cuda())

    # outputs_ for get argmax idx
    if isinstance(outputs, tuple):
        outputLogits = outputs[0]
    else:
        outputLogits = outputs
    if not isinstance(idxCat, type(None)):
        outputs_ = outputLogits[:,idxCat]
        if isinstance(outputs, tuple):
            outputs = (outputs_,) + outputs[1:]
    else:
        outputs_ = outputLogits

    # get argmax idx
    if predict_DataAug:
        outputsIdx = F.softmax(outputs_, dim=1).sum(dim=0).argmax(dim=0)
    else:
        outputsIdx = outputs_.argmax(dim=1)
    if not isinstance(idxCat, type(None)):
        outputsIdx = idxCat[outputsIdx]
    outputsIdx = outputsIdx.reshape(-1)

    loss_batch, accu_batch, loss_info = None, None, None
    if not predict:
        loss_ret = criterion(outputs, targetsIdx.squeeze())
        if isinstance(loss_ret, tuple):
            loss, loss_info = loss_ret
        else:
            loss = loss_ret
            loss_info = 0
        if torch.isnan(loss).any():
            raise RuntimeError('loss explose')
        if training:
            loss.backward()
            optimizer.step()
        loss_batch = loss.item()
        accu_batch = outputsIdx.eq(targetsIdx.squeeze()).sum().item()/batch_size
    return outputs, outputsIdx, loss_batch, accu_batch, batch_size, loss_info


# To beadapted to the real need
def trainer_epoch(
    model, dataloader, criterion,
    optimizer=None, training=False,
    predict=False,predict_DataAug=False,
    idxCat=None,
):
    if training:
        model.train()
    else:
        model.eval()
    avg_mtr = AverageMeter(NVars=3)
    lst_outputs = []

    for ith_iter, data_batch in enumerate(dataloader):
        if predict_DataAug:
            print('%d/%d'%(ith_iter, len(dataloader)))
        outputs,outputsIdx,\
        loss_batch, accu_batch, \
        batch_size, loss_info = trainer_batch(
            model, data_batch, criterion,
            training=training, optimizer=optimizer,
            predict=predict, predict_DataAug=predict_DataAug,
            idxCat=idxCat,
        )
        if predict:
            # predict_DataAug
            lst_outputs.append(outputsIdx.cpu().detach().numpy())
        else:
            avg_mtr.update((loss_batch, accu_batch, loss_info), n=batch_size)

    if predict:
        arr_outputs_prd = np.concatenate(lst_outputs)
        return arr_outputs_prd
    else:
        loss_avg,accu_avg,loss_info = avg_mtr.avg()
        print('TriHardLoss: %f'%loss_info)
        return loss_avg, accu_avg



def trainer_epoch_fWGAN(
    ConvNet, G, D,
    dataloader, lst_tsClassEmb,
    criterion,
    optimizer_G=None, optimizer_D=None,
    training=False,NCritics=5,
    stats_G=None, stats_D=None,
):
    lst_feat_g_valid, lst_feat_r_valid = [], []
    ConvNet.eval()
    G.eval(), D.eval()
    for ith_iter, data_batch in enumerate(dataloader):
        # print('%d/%d'%(ith_iter, len(dataloader)))
        if training:
            G.eval(), D.train()
        # D step
        if training:
            optimizer_D.zero_grad()
        imgs, targetsIdx=data_batch
        batch_size = targetsIdx.shape[0]
        imgs, targetsIdx = imgs.cuda(), targetsIdx.cuda()
        embeds_c = [ emb[targetsIdx.squeeze(dim=1), :] for emb in lst_tsClassEmb]
        feats_r = ConvNet(imgs, ReturnFeat=True)
        feats_g = G(embeds_c)
        loss_D,feat_g_valid,feat_r_valid,GP = criterion(feats_g, feats_r, embeds_c, D=True)
        if training:
            loss_D.backward()
            optimizer_D.step()
        if stats_D:
            stats_D.update(
                (loss_D.item(),feat_g_valid.mean().item(),feat_r_valid.mean().item(), GP.mean().item() ),
                n=batch_size, )
        lst_feat_g_valid.append(feat_g_valid.cpu().detach().numpy())
        lst_feat_r_valid.append(feat_r_valid.cpu().detach().numpy())
        # G step
        if training and ith_iter%NCritics==0 and ith_iter!=0:
            # print('******************* Train G *******************')
            G.train(), D.eval()
            optimizer_G.zero_grad()
            feats_g = G(embeds_c)
            loss_G,feat_g_valid,loss_cls = criterion(feats_g, embeds_c, targetsIdx, G=True)
            loss_G.backward()
            optimizer_G.step()
            if stats_G:
                stats_G.update(
                    (loss_G.item(),loss_cls.mean().item()),
                    n=batch_size)
    arr_feat_r_valid = np.concatenate(lst_feat_r_valid)
    arr_feat_g_valid = np.concatenate(lst_feat_g_valid)
    return stats_G, stats_D, arr_feat_g_valid, arr_feat_r_valid
