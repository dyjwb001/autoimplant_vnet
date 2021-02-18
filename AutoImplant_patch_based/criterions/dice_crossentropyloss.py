import torch
import torch.nn as nn


class Dice_CrossentropyLoss(nn.Module):
    def __init__(self):
        super(Dice_CrossentropyLoss, self).__init__()

    def forward(self, input, target):
        #diceloss
        smooth = 1

        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat*target_flat).sum()

        isum = torch.sum(input_flat*input_flat)
        tsum = torch.sum(target_flat*target_flat)

        dloss = (2 * intersection + smooth) / (isum + tsum + smooth)
        dloss = 1 - dloss
        #crossentropy
        cross = nn.CrossEntropyLoss()
        closs = cross(input,target)

        return dloss+closs


