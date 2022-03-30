import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion=nn.ReLU()

    def __call__(self,outputs,is_real,is_disc=None):
        if is_disc:
            if is_real:
                outputs=-outputs
            return self.criterion(1 + outputs).mean()
        else:
            return (-outputs).mean()