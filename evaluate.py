import numpy as np
import torch
from dice import *
from torch import nn

class Total_loss:
    def __init__(self):
        self.CEloss = nn.NLLLoss(weight=torch.tensor([0.01,2,10,25,100,3,10,30]).float().cuda())#
        self.DICEloss = MulticlassDiceLoss(weight=torch.tensor([0.01,2,10,25,100,3,10,30]).float().cuda())
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def __call__(self, pred, target):
        target = target.detach()
    #    print(pred.shape)
        celoss = self.CEloss(torch.log(pred+1e-12), target)
        diceloss = self.DICEloss(pred, target)
        return celoss+diceloss, celoss, diceloss

def cal_iou(pred, target, class_num=8):
    shape = (class_num, pred.size(0), pred.size(1))
    pred.unsqueeze_(0)
    target.unsqueeze_(0)
    pred_onehot = torch.zeros(shape).long().cuda()
    pred_onehot.scatter_(dim=0, index=pred.long(), src=torch.ones(shape).long().cuda())
    target_onehot = torch.zeros(shape).long().cuda()
    target_onehot.scatter_(dim=0, index=target.long(), src=torch.ones(shape).long().cuda())
    smooth = 1e-12
    iou = 0
 #   print(pred_onehot.shape, target_onehot.shape)
    for i, (pred_chip, target_chip) in enumerate(zip(pred_onehot, target_onehot)):
        if i != 0:
 #            print(pred_chip.shape)
             tp = torch.sum(pred_chip+target_chip == 2).item()
             p = torch.sum(pred_chip == 1).item()
             t = torch.sum(target_chip == 1).item()
#             print(p, t, tp)
             single_iou = (tp+smooth)/(p+t-tp+smooth)
             iou += single_iou
#    print(i)
    return iou/i


