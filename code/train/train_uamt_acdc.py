"""ACDC: total 1356 samples; 30 samples for vadilation;
57 iterations per epoch; max epoch: 527.
UAMT: Uncertainty-Aware Mean Teacher
"""
import argparse
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from val_2D import test_all_case_2D
from utils import losses, ramps

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../../data/ACDC', help='Data root path')
    parser.add_argument('--data_name', type=str,
                        default='ACDC', help='Data name')  
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name, select: unet')
    parser.add_argument('--exp', type=str,
                        default='UAMT', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='MAAGfold', help='cross validation fold')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--ES_interval', type=int,
                        default=10000, help='maximum iteration iternal for early-stopping')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input. Specially, [224, 224] for swinunet')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    args = parser.parse_args()
    return args


def sample_transform():
    rot_k  = random.randrange(0, 4)
    hflip  = random.random() < 0.5
    vflip  = random.random() < 0.5
    scale  = 0.8 + 0.4 * random.random()
    return rot_k, hflip, vflip, scale

def apply_T_to_img(x, rot_k, hflip, vflip, scale):
    # x: [B,1,H,W], float
    B, C, H, W = x.shape
    # scale (bilinear)
    if scale != 1.0:
        Hs = max(1, int(round(H * scale)))
        Ws = max(1, int(round(W * scale)))
        x = F.interpolate(x, size=(Hs, Ws), mode='bilinear', align_corners=False)
    # flips
    if hflip: x = torch.flip(x, dims=[3])
    if vflip: x = torch.flip(x, dims=[2])
    # rotate 90°×k
    if rot_k > 0: x = torch.rot90(x, k=rot_k, dims=[2, 3])
    return x

def apply_T_to_logits(logits, rot_k, hflip, vflip, scale):
    # logits: [B,C,H,W], bilinear cho feature/logits
    B, Cc, H, W = logits.shape
    if scale != 1.0:
        Hs = max(1, int(round(H * scale)))
        Ws = max(1, int(round(W * scale)))
        logits = F.interpolate(logits, size=(Hs, Ws), mode='bilinear', align_corners=False)
    if hflip: logits = torch.flip(logits, dims=[3])
    if vflip: logits = torch.flip(logits, dims=[2])
    if rot_k > 0: logits = torch.rot90(logits, k=rot_k, dims=[2, 3])
    return logits

def apply_T_to_label(y, rot_k, hflip, vflip, scale):
    # y: [B,H,W], long; nearest cho nhãn
    B, H, W = y.shape
    yf = y.float().unsqueeze(1)  # [B,1,H,W]
    if scale != 1.0:
        Hs = max(1, int(round(H * scale)))
        Ws = max(1, int(round(W * scale)))
        yf = F.interpolate(yf, size=(Hs, Ws), mode='nearest')
    if hflip: yf = torch.flip(yf, dims=[3])
    if vflip: yf = torch.flip(yf, dims=[2])
    if rot_k > 0: yf = torch.rot90(yf, k=rot_k, dims=[2, 3])
    return yf.squeeze(1).long()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * ramps.sigmoid_rampup(epoch, 60)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def create_model(ema=False):
    # Network definition
    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=4)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def train(args, snapshot_path):

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    batch_size = args.batch_size
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    ES_interval = args.ES_interval
    ema_decay       = 0.99                      # paper dùng 0.99
    T_MC            = 8                         # số lần MC-Dropout
    C               = num_classes               # cho entropy (C=2)
    lnC             = float(np.log(C))

    # Create model
    model = create_model()
    ema_model = create_model(ema=True)

    # create Dataset
    db_train = BaseDataSets( base_dir=args.root_path, split="train", transform=transforms.Compose(
                            [RandomGenerator(args.patch_size)]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Data loader
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))

    best_performance = 0.0

    # Training
    model.train()
    ema_model.train()

    def gaussian_rampup(t, t_max):
        # lambda(t) = 0.1 * exp(-5 * (1 - t/t_max)^2) (theo paper)
        return 0.1 * np.exp(-5.0 * (1.0 - float(t) / float(max(1, t_max)))**2)

    def tau_threshold(t, t_max):
        # ramp từ 0.75*lnC -> lnC (dùng cùng dạng gaussian cho mượt)
        base = 0.75 * lnC
        return base + (lnC - base) * np.exp(-5.0 * (1.0 - float(t) / float(max(1, t_max)))**2)
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for iter, sampled_batch in enumerate(trainloader):

            img, label = sampled_batch['image'], sampled_batch['label']
            img, label = img.cuda(), label.cuda()
            rot_k, hflip, vflip, scale = sample_transform()
            student_logits = model(img) 
            student_logits_T = apply_T_to_logits(student_logits, rot_k, hflip, vflip, scale)
            label_T = apply_T_to_label(label, rot_k, hflip, vflip, scale)
            # Supervised loss on scribbled pixels after T ----
            loss_ce = ce_loss(student_logits_T, label_T)
            supervised_loss = loss_ce

            # Teacher MC-Dropout
            with torch.no_grad():
                # Chuẩn bị input cho teacher (ảnh + noise nhẹ)
                img_T = apply_T_to_img(img, rot_k, hflip, vflip, scale)  # [B,1,H',W']
                B, C1, Ht, Wt = img_T.shape
                # gom logits qua T_MC lần
                mc_probs = []
                for i in range(T_MC):
                    noise = torch.clamp(torch.randn_like(img_T) * 0.1, -0.2, 0.2)
                    logits_t = ema_model(img_T + noise)                  # [B,C,H',W'] với dropout on
                    probs_t  = F.softmax(logits_t, dim=1)
                    mc_probs.append(probs_t)
                # mean probs
                teacher_mean_probs = torch.stack(mc_probs, dim=0).mean(dim=0)  # [B,C,H',W']

                # predictive entropy (uncertainty)
                entropy = - torch.sum(teacher_mean_probs * torch.log(teacher_mean_probs + 1e-6), dim=1, keepdim=True)  # [B,1,H',W']

            # Consistency loss: MSE giữa probs(student_T) và mean_probs(teacher_T)
            student_probs_T = F.softmax(student_logits_T, dim=1)                  # [B,C,H',W']
            consistency_dist = (student_probs_T - teacher_mean_probs).pow(2).sum(dim=1, keepdim=True)

            # ngưỡng τ ramp-up từ 0.75 lnC -> lnC
            tau = tau_threshold(iter_num, max_iterations)
            mask = (entropy < tau).float()                               

            # trung bình trên vùng mask (tránh chia 0)
            denom = torch.clamp(mask.sum(), min=1.0)
            consistency_loss = (mask * consistency_dist).sum() / denom

            # ---- 5) Tổng loss với Gaussian ramp-up lambda(t) ----
            consistency_weight = gaussian_rampup(iter_num, max_iterations)
            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            # Validation
            if iter_num > 0 and iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss : %f' 
                    %(iter_num, loss.item()))
                
                model.eval()
                metric_list = test_all_case_2D(valloader, model, args)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i], iter_num)
             
                if metric_list[:, 0].mean() > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = metric_list[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', metric_list[:, 0].mean(), iter_num)
                logging.info("avg_metric:{} ".format(metric_list))
                logging.info('iteration %d : dice_score : %f ' % (iter_num, metric_list[:, 0].mean()))

                model.train()


            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num - fresh_iter_num >= ES_interval:
                logging.info("early stooping since there is no model updating over 1w \
                    iteration, iter:{} ".format(iter_num))
                break

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations or (iter_num - fresh_iter_num >= ES_interval):
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    args = parse_args()
    snapshot_path = "../../checkpoints/{}_{}".format(args.data_name, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, os.path.join(snapshot_path, run_id + "_" + os.path.basename(__file__))
    )

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(snapshot_path+"/train_log.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s')) 
    logger.addHandler(console_handler)
    logger.info(str(args))
    start_time = time.time()
    train(args, snapshot_path)
    time_s = time.time()-start_time
    logging.info("time cost: {} s, i.e, {} h".format(time_s,time_s/3600))