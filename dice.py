from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-12

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat
        #print(intersection.shape)
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        #print(loss.shape)
        loss = 1 - loss.sum() / N

        return loss

class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self, weight=None):
        super(MulticlassDiceLoss, self).__init__()
        self.dice = DiceLoss()
        self.weight = weight

    def forward(self, input, target):
        target = target.unsqueeze(1)
 
        target_onehot = torch.zeros_like(input).long()
        target_onehot.scatter_(dim=1, index=target.long(), src=torch.ones_like(input).long())
        C = target_onehot.shape[1]
        #print(input.shape)
        #print(torch.unique(target_onehot.sum(1)), torch.unique(input.sum(1)))
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        totalLoss = 0

        for i in range(1, C):
            diceLoss = self.dice(input[:, i, :, :], target_onehot[:, i, :, :])
            if self.weight is not None:
                diceLoss *= self.weight[i]
            totalLoss += diceLoss
        
        return totalLoss/(C-1)
