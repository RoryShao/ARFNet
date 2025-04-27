#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

"""
implement by SHOT with kmeans
"""
"""
implement by SHOT
"""
def obtain_label(loader, netF, netB, netC, args, log):
    start_test = True    #
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test) 
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            distribution = netF(inputs)
            feas = netB(distribution)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)   # [498, 31]
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()   # [498, 257]
    K = all_output.size(1)    # K = 31
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)   # 31 x 256
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log.logger.info('Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100))
    return pred_label.astype('int')




"""
implement by dynamic centroid
"""
def obtain_label_multi_centriods(loader, netF, netB, netC, args, log):
    start_test = True    #
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            logits = netF(inputs)
            feas = netB(logits)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = torch.exp(all_output)   
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()   # [498, 257]
    K = all_output.size(1)    # K = 31
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)   # 31 x 256
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    initc_ema = initc
    for round in range(5):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        initc_ema = initc_ema * args.lamb + (1-args.lamb)*initc 

        dd = cdist(all_fea, initc_ema[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log.logger.info('Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100))
    return pred_label.astype('int')



def obtain_domainnet_label_multi_centriods(loader, netF, netB, netC, args, log):
    start_test = True    #
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data, labels, indexs = next(iter_test)
            inputs = data[0]
            labels = labels
            inputs = inputs.cuda()
            logits = netF(inputs)
            feas = netB(logits)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = torch.exp(all_output)   # [498, 31]
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()   # [498, 257]
    K = all_output.size(1)    # K = 31
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)   # 31 x 256
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    initc_ema = initc
    for round in range(5):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        initc_ema = initc_ema * args.lamb + (1-args.lamb)*initc 

        dd = cdist(all_fea, initc_ema[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log.logger.info('Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100))
    return pred_label.astype('int')





if __name__ == '__main__':
    pass
