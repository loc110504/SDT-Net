import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import torch.nn.functional as F
def Jigsaw(imgs, num_x, num_y, shuffle_index=None):
    split_w, split_h = int(imgs.shape[2] / num_x), int(imgs.shape[3] / num_y)
    out_imgs = torch.zeros_like(imgs)
    imgs = imgs.unsqueeze(0)
    # -----------------------
    # 分块
    # -----------------------
    patches = torch.split(imgs, split_w, dim=3)
    patches = [torch.split(p, split_h, dim=4) for p in patches]
    patches = torch.cat([torch.cat(p, dim=0) for p in patches], dim=0)
    # -----------------------
    # shuffle_index为空则打乱, 否则还原
    # -----------------------
    if shuffle_index is None:
        shuffle_index = np.random.permutation(num_x * num_y)
    else:
        shuffle_index = list(shuffle_index)
        shuffle_index = [shuffle_index.index(i) for i in range(num_x * num_y)]
    patches = patches[shuffle_index]
    # -----------------------
    # 拼接
    # -----------------------
    x_index, y_index = 0, 0
    for patch in patches:
        out_imgs[:, :, y_index:y_index + split_h, x_index:x_index + split_w] = patch
        x_index += split_w
        if x_index == out_imgs.shape[2]:
            x_index = 0
            y_index += split_h
    return out_imgs, shuffle_index


def RandomBrightnessContrast(img, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
    output = torch.zeros_like(img)
    threshold = 0.5

    for i in range(output.shape[0]):
        img_min, img_max = torch.min(img[i]), torch.max(img[i])

        output[i] = (img[i] - img_min) / (img_max - img_min) * 255.0
        if random.random() < p:
            brightness = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            output[i] = torch.clamp(output[i] * brightness, 0., 255.)

            contrast = 0.0 + random.uniform(-contrast_limit, contrast_limit)
            output[i] = torch.clamp(output[i] + (output[i] - threshold * 255.0) * contrast, 0., 255.)

        output[i] = output[i] / 255.0 * (img_max - img_min) + img_min
    return output


def Cutout_max(imgs, labels, device, n_holes=1):
    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []

    for m in range(num):
        label = labels[m, :, :, :]
        middle_channels = label[1:4, :, :]
        tensor = torch.sum(middle_channels, dim=0, keepdim=True)
        summed_tensor = torch.sum(tensor, dim=0, keepdim=True)
        # 找到非零元素的索引
        non_zero_indices = torch.nonzero(summed_tensor)
        if non_zero_indices.numel() == 0:
            img = imgs[m, :, :, :]
            imgs_list.append(img)
            labels_list.append(label)
        else:
            # 分别找到最小的行、列和最大的行、列
            min_coords = torch.min(non_zero_indices, dim=0)[0]  # 最小的坐标
            max_coords = torch.max(non_zero_indices, dim=0)[0]  # 最大的坐标

            y3 = min_coords.cpu()[1].item()
            x3 = min_coords.cpu()[2].item()
            y4 = max_coords.cpu()[1].item()
            x4 = max_coords.cpu()[2].item()
            x = (x3 + x4) // 2
            y = (y3 + y4) // 2
            chang = (x4 - x3) // 1
            kuan = (y4 - y3) // 1
            chang = int(chang * 1.2)
            kuan = int(kuan * 1.2)
            if chang % 2 != 0:
                chang = chang + 1
            if kuan % 2 != 0:
                kuan = kuan + 1

            img = imgs[m, :, :, :]
            new_img = img.clone()
            # 创建初始的 One-Hot 编码
            new_label = label.clone()
            # new_label = torch.zeros((5, h, w), device=device)  # 假设有 5 个类别
            mask = np.ones((1, h, w), np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.to(device)


            for n in range(n_holes):
                y1 = np.clip(y - kuan // 2, 0, h)
                y2 = np.clip(y + kuan // 2, 0, h)
                x1 = np.clip(x - chang // 2, 0, w)
                x2 = np.clip(x + chang // 2, 0, w)

                new_img[:, :, :] = 0
                new_img[0, y1: y2, x1: x2] = img[0, y1: y2, x1: x2]
                # 更新 One-Hot 编码
                new_label[:, y1:y2, x1:x2] = 0
                new_label[:, y1:y2, x1:x2] = label[:, y1:y2, x1:x2]

            # mask = mask.expand_as(img)
            # img = img * mask
            # label = label * mask
            imgs_list.append(new_img)
            labels_list.append(new_label)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)

    return imgs_out, labels_out
def Cutout_min(imgs, labels, device, n_holes=1):
    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []

    for m in range(num):
        label = labels[m, :, :, :]
        middle_channels = label[1:4, :, :]
        tensor = torch.sum(middle_channels, dim=0, keepdim=True)
        summed_tensor = torch.sum(tensor, dim=0, keepdim=True)
        # 找到非零元素的索引
        non_zero_indices = torch.nonzero(summed_tensor)
        if non_zero_indices.numel() == 0:
            img = imgs[m, :, :, :]
            imgs_list.append(img)
            labels_list.append(label)
        else:
            # 分别找到最小的行、列和最大的行、列
            min_coords = torch.min(non_zero_indices, dim=0)[0]  # 最小的坐标
            max_coords = torch.max(non_zero_indices, dim=0)[0]  # 最大的坐标

            y3 = min_coords.cpu()[1].item()
            x3 = min_coords.cpu()[2].item()
            y4 = max_coords.cpu()[1].item()
            x4 = max_coords.cpu()[2].item()
            x = (x3 + x4) // 2
            y = (y3 + y4) // 2
            chang = (x4 - x3) // 1
            kuan = (y4 - y3) // 1
            chang = int(chang * 0.8)
            kuan = int(kuan * 0.8)
            if chang % 2 != 0:
                chang = chang + 1
            if kuan % 2 != 0:
                kuan = kuan + 1

            img = imgs[m, :, :, :]
            new_img = img.clone()
            # 创建初始的 One-Hot 编码
            new_label = label.clone()
            # new_label = torch.zeros((5, h, w), device=device)  # 假设有 5 个类别
            mask = np.ones((1, h, w), np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.to(device)


            for n in range(n_holes):
                y1 = np.clip(y - kuan // 2, 0, h)
                y2 = np.clip(y + kuan // 2, 0, h)
                x1 = np.clip(x - chang // 2, 0, w)
                x2 = np.clip(x + chang // 2, 0, w)

                mask[0, y1: y2, x1: x2] = 0.
                new_img[0, y1: y2, x1: x2] = 0.
                # 更新 One-Hot 编码
                new_label[:, y1:y2, x1:x2] = 0
                new_label[4, y1:y2, x1:x2] = 1  # 更新为未标注类别

            # mask = mask.expand_as(img)
            # img = img * mask
            # label = label * mask
            imgs_list.append(new_img)
            labels_list.append(new_label)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)

    return imgs_out, labels_out


class DynamicTripletLoss(nn.Module):
    def __init__(self, initial_margin=0.5, max_margin=2.0, epochs=100):
        super(DynamicTripletLoss, self).__init__()
        self.initial_margin = initial_margin
        self.max_margin = max_margin
        self.epochs = epochs
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=self.initial_margin, p=2)  # 使用L2距离

    def update_margin(self, current_epoch):
        # 动态调整阈值，随着训练轮次逐渐增大
        margin = self.initial_margin + (self.max_margin - self.initial_margin) * (current_epoch / self.epochs)
        self.triplet_loss_fn.margin = margin
        return margin

    def forward(self, anchor, positive, negative, current_epoch):
        # 更新阈值
        margin = self.update_margin(current_epoch)
        # print(f"Epoch {current_epoch}, Margin: {margin}")

        # 计算三元组损失
        loss = self.triplet_loss_fn(anchor, positive, negative)
        return loss

def soft_erode(img):
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)

def exrct_boundary(img, iter_):
    # img = F.softmax(img, dim=1)

    img1 = img.clone()
    for j in range(iter_):
        img1 = soft_erode(img1)
    boundary = F.relu(img - img1)  # 计算边界

    # 二值化：将非零值设为1，零值保持为0
    # boundary_binary = (boundary > 0.5).float()

    return boundary

class BoundaryLoss(nn.Module):
    def __init__(self, iter_=1, weight_boundary=0.03):
        super(BoundaryLoss, self).__init__()
        self.iter_ = iter_
        self.weight_boundary = weight_boundary

    def forward(self, pred_boundaries, true_boundaries):
        """
        计算边界损失
        - pred_boundaries: 模型预测边界，通常是经过边界提取操作的
        - true_boundaries: 真实标签的边界，通常也是经过边界提取操作的
        """

        # 计算交集与联合（类似于Dice损失）
        intersection = torch.sum(pred_boundaries * true_boundaries)
        union = torch.sum(pred_boundaries) + torch.sum(true_boundaries)
        boundary_loss = 1 - (2 * intersection + 1e-5) / (union + 1e-5)

        # 乘上权重
        return self.weight_boundary * boundary_loss


















