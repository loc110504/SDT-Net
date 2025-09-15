import argparse
import logging
import os
import random
import shutil
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

from utils.util import process_pseudo_label
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.mscmr import MSCMRDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.Jigsaw import Cutout_min, Jigsaw, RandomBrightnessContrast, exrct_boundary, BoundaryLoss
from val import test_single_volume_scribblevs

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/MSCMR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='TABNet', help='experiment_name')
parser.add_argument('--data', type=str,
                    default='MSCMR', help='experiment_name')
parser.add_argument('--tau', type=float,
                    default=0.5, help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='MAAGfold70', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def create_model(ema=False,num_classes=4):
    # Network definition
    net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    model = create_model(ema=False,num_classes=4)

    db_train = MSCMRDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), sup_type=args.sup_type)
    db_val = MSCMRDataSets(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    bd_loss_fn = BoundaryLoss(iter_=1, weight_boundary=1.0)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 1.0
    K = 5
    # hyperparams theo paper
    lambda1 = 1.0  # TAS (CE nhánh i, j, k)
    lambda2 = 0.3  # PL (Dice consistency với y_pl)
    lambda3 = 0.1  # Boundary-aware loss

    for epoch_num in iterator:
        for i_batch, sampled in enumerate(trainloader):

            # Get data
            image = sampled['image'].cuda()   # (B,1,H,W)
            scrib = sampled['label'].cuda()   # (B,H,W) integer; ignore_index=4
            if scrib.dtype != torch.long:
                scrib = scrib.long()

            # 3 Augment
            # Cutout
            scrib = scrib.cuda(non_blocking=True)  
            scrib_oh = F.one_hot(scrib.clamp(0, K-1), num_classes=K)          # (B,H,W,K)
            scrib_oh = scrib_oh.permute(0, 3, 1, 2).contiguous().float()
            img_i, labels_i_oh = Cutout_min(
                imgs=image,
                labels=scrib_oh,
                device=image.device,
                n_holes=1
            ) 
            scrib_i = torch.argmax(labels_i_oh, dim=1)
            # Jigsaw
            img_j, shuffle_idx = Jigsaw(image, num_x=2, num_y=2, shuffle_index=None)
            # Intensity
            img_k = RandomBrightnessContrast(image)
            logits_i = model(img_i)   # (B,C,H,W)
            logits_j_shuf = model(img_j)
            logits_k = model(img_k)
            logits_j, _ = Jigsaw(logits_j_shuf, num_x=2, num_y=2, shuffle_index=shuffle_idx)

            y_i = torch.softmax(logits_i, dim=1)
            y_j = torch.softmax(logits_j, dim=1)
            y_k = torch.softmax(logits_k, dim=1)

            # TAS loss
            loss_ce_i = ce_loss(logits_i, scrib)
            loss_ce_j = ce_loss(logits_j, scrib)
            loss_ce_k = ce_loss(logits_k, scrib)
            loss_TAS = loss_ce_i + loss_ce_j + loss_ce_k
            # BAP
            with torch.no_grad():
                denom = (loss_ce_j.detach() + loss_ce_k.detach()).clamp_min(1e-8)
                w_j = (loss_ce_k.detach() / denom).item()
                w_k = (loss_ce_j.detach() / denom).item()
            mixed_prob = w_j * y_j + w_k * y_k 
            y_pl = torch.argmax(mixed_prob.detach(), dim=1)  
            loss_PL = dice_loss(y_j, y_pl.unsqueeze(1)) + dice_loss(y_k, y_pl.unsqueeze(1))

            y_pl_oh = F.one_hot(y_pl, num_classes=num_classes).permute(0, 3, 1, 2).float()
            B_pl = exrct_boundary(y_pl_oh, iter_=1)
            B_j  = exrct_boundary(y_j,  iter_=1)
            B_k  = exrct_boundary(y_k,  iter_=1)    
            loss_BD = bd_loss_fn(B_j, B_pl.detach()) + bd_loss_fn(B_k, B_pl.detach())

            # Total loss
            loss = lambda1 * loss_TAS + lambda2 * loss_PL + lambda3 * loss_BD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("train/loss_total", loss.item(), iter_num)
            writer.add_scalar("train/loss_TAS_ce", loss_TAS.item(), iter_num)
            writer.add_scalar("train/loss_PL_dice", loss_PL.item(), iter_num)
            writer.add_scalar("train/loss_BD", loss_BD.item(), iter_num)

            if iter_num % 400 == 0:
                logging.info(
                'iteration %d : loss : %f, loss_TAS: %f, loss_PL: %f, loss_BD: %f' 
                % (iter_num, loss.item(), loss_TAS.item(), loss_PL.item(), loss_BD.item()))

            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_scribblevs(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num > 0 and iter_num % 500 == 0:
                if alpha > 0.01:
                    alpha = alpha - 0.01
                else:
                    alpha = 0.01

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../../checkpoints/{}_{}".format(args.data, args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)