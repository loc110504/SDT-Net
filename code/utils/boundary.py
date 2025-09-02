# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:18:43 2024

@author: ZHANGXIAO
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1),(1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

def exrct_boundary(img, iter_):
    img1 = img.clone()
    for j in range(iter_):
        img = soft_erode(img)
    return F.relu(img1-img)


if __name__ == '__main__':

    path = 'D:/File/Paper/Scribble_Project/CycleMix-Jigsaw-Edge/util/individualImage (1).png'

    img = np.array(Image.open(path))[:,:,0] / 255.
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    prob_map = torch.cat((1-img_tensor, img_tensor), dim=1)

    # 设置温度参数
    tau = 0.1  # 温度超参数，控制离散性
    hard = False  # 是否执行硬选择（argmax）

    output = F.softmax(prob_map / tau, dim=1)
    I = Image.fromarray(output[0, 1, :, :].cpu().numpy() * 255.)
    I.show()

    ### 提取boundary区域
    prob_boundary = exrct_boundary(output[:, 1, :, :].unsqueeze(0), iter_=1)
    prob_bound_squeezed = prob_boundary.squeeze(0).squeeze(0)
    prob_boundArr = prob_bound_squeezed.cpu().numpy() * 255.
    I = Image.fromarray(prob_boundArr)
    I.show()














