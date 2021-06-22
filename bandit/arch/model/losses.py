import torch.nn.functional as F
import torch


def MSELoss(output, target):
    return F.mse_loss(output, target)
