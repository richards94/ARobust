import torch
from torch import Tensor, nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ARobustLoss(nn.Module):
    def __init__(self, alpha=1, c=1):
        super(ARobustLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, y_true, y_pred):
        alpha = self.alpha
        c = self.c
        x = y_true - y_pred
        absolute_x = torch.abs(x)
        exponent_part = torch.pow(absolute_x / c, 2.1 - y_true)
        absolute_alpha = abs(alpha-2)
        loss = absolute_alpha / alpha * (torch.pow(exponent_part / absolute_alpha + 1, alpha / 2) - 1)
        loss = torch.mean(torch.sum(loss, axis=[1, 2, 3]), axis=0)
        return loss


class RobustLoss(nn.Module):
    def __init__(self, alpha=1, c=1):
        super(RobustLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, y_true, y_pred):
        alpha = self.alpha
        c = self.c
        x = y_true - y_pred
        absolute_x = torch.abs(x)
        exponent_part = torch.pow(absolute_x / c, 2)
        absolute_alpha = abs(alpha-2)
        loss = absolute_alpha / alpha * (torch.pow(exponent_part / absolute_alpha + 1, alpha / 2) - 1)
        loss = torch.mean(torch.sum(loss, axis=[1, 2, 3]), axis=0)
        return loss