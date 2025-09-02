"""ACDC: total 1356 samples; 30 samples for validation;
57 iterations per epoch; max epoch: 527.
This file stores code implementation for ShapePU
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ----------------------------
# ShapePU utilities
# ----------------------------
def get_logits(model, x):
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out

def softmax_probs(logits):
    return torch.softmax(logits, dim=1)

def sample_spatial_transform():
    """Random 90° rot + flips; trả về hàm apply T(.) để dùng nhất quán cho ảnh/nhãn/mask."""
    k = random.randint(0, 3)          # 0,1,2,3 * 90 deg
    flip_h = random.random() < 0.5
    flip_v = random.random() < 0.5
    def apply(t):
        # t: [B,C,H,W] hoặc [B,1,H,W]
        if k:
            t = torch.rot90(t, k, dims=(-2, -1))
        if flip_h:
            t = torch.flip(t, dims=(-1,))
        if flip_v:
            t = torch.flip(t, dims=(-2,))
        return t
    return apply

def random_square_mask(x, cutout_size):
    """Mask nhị phân z (B,1,H,W), 0 trong ô cutout; 1 ngoài vùng cắt."""
    B, _, H, W = x.shape
    z = torch.ones((B,1,H,W), device=x.device, dtype=x.dtype)
    if cutout_size <= 0:
        return z
    ch = min(cutout_size, H-1)
    cw = min(cutout_size, W-1)
    for b in range(B):
        if H - ch <= 0 or W - cw <= 0:
            continue
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        z[b, :, top:top+ch, left:left+cw] = 0.0
    return z

def mask_labels_with_cutout_and_transform(labels, z, apply_T, ignore_index):
    """
    Nhãn gốc labels: (B,H,W).
    Trả về labels' cho X' = T(z ⊙ X):
      - biến đổi labels cùng T;
      - set ignore_index tại vùng z==0 sau biến đổi.
    """
    lbl = labels.unsqueeze(1).float()
    lbl_t = apply_T(lbl).squeeze(1).long()
    z_t = apply_T(z)
    lbl_t[z_t.squeeze(1)==0] = ignore_index
    return lbl_t

def cosine_loss_prob(a, b, eps=1e-8):
    """Cosine distance giữa phân phối xác suất (B,C,H,W) → scalar."""
    a = a.flatten(1)
    b = b.flatten(1)
    a = a / (a.norm(dim=1, keepdim=True) + eps)
    b = b / (b.norm(dim=1, keepdim=True) + eps)
    return (1.0 - (a*b).sum(dim=1)).mean()

@torch.no_grad()
def estimate_pl_prior_from_labeled(labels, num_classes, ignore_index):
    """
    p_l(c_j): tần suất lớp trên pixel có nhãn trong batch hiện tại.
    Trả về tensor (C,) chuẩn hoá tổng = 1. Nếu thiếu lớp → gán epsilon.
    """
    C = num_classes
    eps = 1e-6
    counts = labels.new_zeros(C, dtype=torch.float)
    mask = (labels != ignore_index)
    total = mask.sum().item()
    if total == 0:
        return torch.full((C,), 1.0/C, device=labels.device)
    for j in range(C):
        counts[j] = torch.sum((labels == j)).float()
    pl = counts / max(total, 1.0)
    pl = torch.clamp(pl, min=eps)
    pl = pl / pl.sum()
    return pl

def prior_adjust_posteriors(p_l_post, p_u_prior, p_l_prior, eps=1e-8):
    """
    p_u(c|x) ∝ (p_u(c)/p_l(c)) * p_l(c|x), rồi chuẩn hoá theo lớp.
    p_l_post: (N,C) từ softmax trên logits
    p_u_prior, p_l_prior: (C,)
    """
    w = (p_u_prior / (p_l_prior + eps)).clamp(min=eps)
    adj = p_l_post * w[None, :]
    adj_sum = adj.sum(dim=1, keepdim=True).clamp(min=eps)
    return adj / adj_sum

@torch.no_grad()
def em_alpha_from_batch(p_soft, labels, num_classes, ignore_index, p_u_prior=None, ema=0.7):
    """
    EM đa lớp trên pixel **unlabeled** của batch:
      - E: p_u(c|x) qua prior-adjustment
      - M: alpha_j = mean_x p_u(c_j|x)
      - Cập nhật prior p_u bằng EMA.
    p_soft: (B,C,H,W), labels: (B,H,W)
    Trả về alpha (C,), p_u_new (C,)
    """
    B, C, H, W = p_soft.shape
    device = p_soft.device
    pl = estimate_pl_prior_from_labeled(labels, C, ignore_index)  # (C,)
    # unlabeled mask
    unlabeled = (labels == ignore_index)  # (B,H,W)
    if unlabeled.sum() == 0:
        # nếu không có unlabeled trong batch → giữ nguyên prior đều
        alpha = torch.full((C,), 1.0/C, device=device)
        if p_u_prior is None:
            p_u_prior = alpha.clone()
        return alpha, p_u_prior

    probs = p_soft.permute(0,2,3,1)[unlabeled]  # (N_u, C)
    if p_u_prior is None:
        p_u_prior = pl.clone()
    post_u = prior_adjust_posteriors(probs, p_u_prior, pl)  # (N_u,C)
    alpha = post_u.mean(dim=0)  # (C,)
    # EMA prior
    p_u_new = ema * p_u_prior + (1.0 - ema) * alpha
    p_u_new = (p_u_new / p_u_new.sum().clamp_min(1e-8)).clamp_min(1e-6)
    return alpha, p_u_new

def negative_loss_unlabeled(p_soft, labels, alpha, foreground_only=True, ignore_index=None):
    """
    L_- = - Σ_j Σ_{x in neg(j)} log p(\bar c_j|x), với neg(j): unlabeled KHÔNG thuộc top α_j của lớp j.
    p_soft: (B,C,H,W), labels: (B,H,W), alpha: (C,)
    """
    B, C, H, W = p_soft.shape
    device = p_soft.device
    unlabeled = (labels == ignore_index)
    if unlabeled.sum() == 0:
        return torch.tensor(0.0, device=device)

    loss = torch.tensor(0.0, device=device)
    classes = range(1, C) if foreground_only else range(C)

    # flatten unlabeled pixels
    for j in classes:
        pj = p_soft[:, j, :, :][unlabeled]  # (N_u,)
        Nu = pj.numel()
        if Nu == 0:
            continue
        k = int(max(0, min(Nu, round(alpha[j].item() * Nu))))
        if k > 0:
            topk_val, topk_idx = torch.topk(pj, k, largest=True, sorted=False)
            neg_mask = torch.ones(Nu, dtype=torch.bool, device=device)
            neg_mask[topk_idx] = False
        else:
            neg_mask = torch.ones(Nu, dtype=torch.bool, device=device)
        p_not_j = (1.0 - pj.clamp(0,1))[neg_mask].clamp_min(1e-8)
        if p_not_j.numel() > 0:
            loss = loss + (-torch.log(p_not_j).mean())
    return loss


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../../data/ACDC', help='Data root path')
    parser.add_argument('--data_name', type=str,
                        default='ACDC', help='Data name')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name, select: unet')
    parser.add_argument('--exp', type=str,
                        default='ShapePU', help='experiment_name')
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
    parser.add_argument('--base_lr', type=float,  default=1e-4,
                        help='segmentation network learning rate (paper ~1e-4)')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input. Specially, [224, 224] for swinunet')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')

    # ---- ShapePU-specific ----
    parser.add_argument('--warmup_epochs', type=int, default=100, help='only L+ and L_global')
    parser.add_argument('--lambda_neg', type=float, default=1.0, help='weight for L_-')
    parser.add_argument('--lambda_global', type=float, default=0.05, help='weight for L_global')
    parser.add_argument('--cutout_size', type=int, default=32, help='square side for cutout')
    parser.add_argument('--em_ema', type=float, default=0.7, help='EMA factor for prior update')
    parser.add_argument('--em_every', type=int, default=0,
                        help='EM update every N iters (0 = once per epoch)')
    args = parser.parse_args()
    return args


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
    ignore_index = num_classes

    # Create model
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes).cuda()
    model_parameter = sum(p.numel() for p in model.parameters())
    logging.info("model_parameter:{}M".format(round(model_parameter / (1024*1024),2)))

    # create Dataset
    db_train = BaseDataSets(
        base_dir=args.root_path, split="train",
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
        fold=args.fold, sup_type=args.sup_type
    )
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

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=ignore_index)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    fresh_iter_num = iter_num
    max_epoch = max_iterations // len(trainloader) + 1
    logging.info("max epoch: {}".format(max_epoch))

    best_performance = 0.0

    # priors for EM (init = uniform)
    p_u_prior = torch.full((num_classes,), 1.0/num_classes, device='cuda')

    # Training
    model.train()
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # flag bật PU sau warmup
        use_pu = (epoch_num >= args.warmup_epochs)
        # nếu đặt em_every=0 → chạy EM 1 lần/epoch
        do_em_this_epoch = True

        for it, sampled_batch in enumerate(trainloader):

            img, label = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()  # img: (B,1,H,W), label: (B,H,W)

            # --- forward gốc ---
            logits = get_logits(model, img)                      # (B,C,H,W)
            probs  = softmax_probs(logits)

            # --- cutout + transform cho global consistency ---
            z = random_square_mask(img, args.cutout_size)        # (B,1,H,W)
            T = sample_spatial_transform()
            img_prime = T(z * img)                               # X' = T(z ⊙ X)
            logits_prime = get_logits(model, img_prime)
            probs_prime = softmax_probs(logits_prime)
            # T(z ⊙ f(X))
            probs_cut_then_T = T(z * probs)

            # --- losses ---
            # 1) L_plus: CE trên pixel có nhãn (ignore unlabeled)
            L_plus = ce_loss(logits, label.long())

            # 2) L_global: cosine distance giữa T(z ⊙ f(X)) và f(T(z ⊙ X))
            L_global = cosine_loss_prob(probs_cut_then_T, probs_prime) * args.lambda_global

            # 3) EM + Negative loss trên unlabeled (sau warmup)
            if use_pu:
                # chạy EM định kỳ
                run_em = (args.em_every == 0 and do_em_this_epoch) or (args.em_every > 0 and (iter_num % args.em_every == 0))
                if run_em:
                    with torch.no_grad():
                        alpha, p_u_prior = em_alpha_from_batch(
                            probs, label, num_classes, ignore_index,
                            p_u_prior=p_u_prior, ema=args.em_ema
                        )
                    do_em_this_epoch = False  # chỉ một lần/epoch nếu em_every=0
                else:
                    # không cập nhật → dùng prior hiện có như alpha gần đúng
                    alpha = p_u_prior

                L_neg = negative_loss_unlabeled(
                    probs, label, alpha,
                    foreground_only=True, ignore_index=ignore_index
                ) * args.lambda_neg
            else:
                L_neg = torch.tensor(0.0, device=logits.device)

            loss = L_plus + L_global + L_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LR poly decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # logs
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/L_plus', L_plus.item(), iter_num)
            writer.add_scalar('info/L_global', L_global.item(), iter_num)
            writer.add_scalar('info/L_neg', L_neg.item() if use_pu else 0.0, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)

            # Validation mỗi 200 iters
            if iter_num > 0 and iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss: %.5f | L+: %.5f | Lg: %.5f | L-: %.5f | usePU: %s'
                    % (iter_num, loss.item(), L_plus.item(), L_global.item(), (L_neg.item() if use_pu else 0.0), str(use_pu))
                )
                model.eval()
                metric_list = test_all_case_2D(valloader, model, args)

                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i], iter_num)

                mean_dice = metric_list[:, 0].mean()
                if mean_dice > best_performance:
                    fresh_iter_num = iter_num
                    best_performance = mean_dice
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score', mean_dice, iter_num)
                logging.info("avg_metric:{} ".format(metric_list))
                logging.info('iteration %d : dice_score : %f ' % (iter_num, mean_dice))
                model.train()

            # Save định kỳ
            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            # Early stop
            if iter_num - fresh_iter_num >= ES_interval:
                logging.info("early stopping since there is no model updating over interval, iter:{} ".format(iter_num))
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
