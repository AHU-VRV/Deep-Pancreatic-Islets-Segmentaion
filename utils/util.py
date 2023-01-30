#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from networks.aanet import AANet

def calculate_Accuracy(confusion):
    if confusion.shape == (1,1):
        Acc = 1.
        IU = [1.,0.]
        Se = 0
        Sp = 1.
        PPV = 0
        meanIU = np.mean(IU)
        F_score = 0
    else:
        confusion=np.asarray(confusion)
        pos = np.sum(confusion, 1).astype(np.float32) # 1 for row
        res = np.sum(confusion, 0).astype(np.float32) # 0 for coloum
        tp = np.diag(confusion).astype(np.float32)
        IU = tp / (pos + res - tp)

        meanIU = np.mean(IU)
        Acc = np.sum(tp) / np.sum(confusion)
        if confusion[1][1]+confusion[1][0] == 0:
            print('confusion:', confusion)
            Se = 0.
            PPV = confusion[1][1] / (confusion[1][1] + confusion[0][1])

        elif confusion[1][1]+confusion[0][1] == 0:
            PPV = 0.
            Se = confusion[1][1] / (confusion[1][1] + confusion[1][0])
        else:
            Se = confusion[1][1] / (confusion[1][1] + confusion[1][0])
            PPV = confusion[1][1] / (confusion[1][1] + confusion[0][1])

        if Se + PPV == 0:
            F_score = 0
        else:
            F_score = (2 * Se * PPV) / (Se + PPV)

        Sp = confusion[0][0] / (confusion[0][0]+confusion[0][1])

    return  meanIU,Acc,Se,Sp,IU,PPV,F_score

def get_model(model_name):
    if model_name=='AANet':
        return AANet




def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network, without sigmoid
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """


    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    #loss_val = - torch.log(1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    #loss_val = - torch.log(1 + torch.exp(-output))
    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss