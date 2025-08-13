import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler

import networks


def process_pseudo_label(outputs_soft_ema,tau=0.5):
    pre_argmax = torch.argmax(outputs_soft_ema,dim=1)
    pseudo_label_0s = outputs_soft_ema[:, 0, :, :]
    pseudo_label_1s = outputs_soft_ema[:, 1, :, :]
    pseudo_label_2s = outputs_soft_ema[:, 2, :, :]
    pseudo_label_3s = outputs_soft_ema[:, 3, :, :]
    a0 = torch.where((pseudo_label_0s > tau) & (pre_argmax == 0) , 5, 0)
    a1 = torch.where((pseudo_label_1s > tau) & (pre_argmax == 1), 1, 0)
    a2 = torch.where((pseudo_label_2s > tau) & (pre_argmax == 2), 2, 0)
    a3 = torch.where((pseudo_label_3s > tau) & (pre_argmax == 3), 3, 0)
    pseudo_label_used = ((a0 + a1 + a2 + a3) >0).to(torch.int32)
    pseudo_label_no_used = torch.ones_like(pseudo_label_used) - pseudo_label_used
    pseudo_label = a0 + a1 + a2 + a3 + 4*pseudo_label_no_used
    pseudo_label[pseudo_label==5]=0
    return pseudo_label




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)