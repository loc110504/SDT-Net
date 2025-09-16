import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR) 

import importlib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader.acdc_scribblevc import BaseDataSets_ScribbleVC, RandomGeneratorVC
from torchmetrics.classification import MultilabelAccuracy
from tool import pyutils
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from utils.losses import pDLoss, SupConLoss
from val_scribblevc import test_single_volume_CAM as test_single_volume, calculate_metric_percase

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='scribbleVC', help='experiment_name')
parser.add_argument('--data', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='MAAGfold70', help='cross validation')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision type')
parser.add_argument('--model', type=str,
                    default='scribbleVC', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=5e-4,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', nargs='+', type=int, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

# Additional ScribbleVC specific arguments
parser.add_argument("--networks", default="networks.scribblevc", type=str)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--wt_dec", default=5e-4, type=float, help='optimizer weight decay')
parser.add_argument("--arch", default='ACDC', type=str)
parser.add_argument("--session_name", default="TransCAM", type=str)
parser.add_argument("--crop_size", default=512, type=int)
parser.add_argument("--pretrain_weights", default='', type=str)
parser.add_argument("--tblog", default='ACDC/scribbleVC', type=str)
parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer name')
parser.add_argument('--lrdecay', action="store_true", help='lr decay')
parser.add_argument('--linear_layer', action="store_true", help='linear layer')
parser.add_argument('--bilinear', action="store_false", help='use bilinear in Upsample layer')
parser.add_argument('--weight_pseudo_loss', type=float, default=0.1, help='pseudo label loss')
parser.add_argument('--weight_crf', type=float, default=0.1, help='crf loss')
parser.add_argument('--weight_cls', type=float, default=0.1, help='cls loss')
parser.add_argument('--temp', type=float, default=0.1, help='temperature for contrastive loss function SupConLoss')
parser.add_argument('--no_class_rep', action="store_true", help='ban class representation')
parser.add_argument("--val_every_epoches", default=1, type=int)
parser.add_argument('--val_mode', action="store_true")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def create_model():
    """Create ScribbleVC model"""
    num_classes = args.num_classes
    model = getattr(importlib.import_module(args.networks), 'scribbleVC_' + args.arch)(
        linear_layer=args.linear_layer,
        bilinear=args.bilinear,
        num_classes=num_classes,
        batch_size=args.batch_size
    )
    
    if len(args.pretrain_weights) != 0:
        logging.info("Load pretrain weight from {}".format(args.pretrain_weights))
        model.load_state_dict(torch.load(args.pretrain_weights), False)
    
    model = model.cuda()
    logging.info('Model is from {}'.format(model.__class__))
    return model


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    # Create model
    model = create_model()
    
    # Create datasets
    db_train = BaseDataSets_ScribbleVC(
        base_dir=args.root_path, 
        split="train", 
        transform=transforms.Compose([RandomGeneratorVC(args.patch_size)]), 
        fold=args.fold, 
        sup_type=args.sup_type
    )
    db_val = BaseDataSets_ScribbleVC(base_dir=args.root_path, fold=args.fold, split="val")

    # Create dataloaders
    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    
    # Create optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.wt_dec, eps=1e-8)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.wt_dec, eps=1e-8)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # Loss definitions
    if args.sup_type == "label":
        ce_loss = CrossEntropyLoss(ignore_index=0)
        dice_loss = pDLoss(num_classes, ignore_index=0)
    elif args.sup_type == "scribble":
        ce_loss = CrossEntropyLoss(ignore_index=4)
        dice_loss = pDLoss(num_classes, ignore_index=4)
    
    gatecrf_loss = ModelLossSemsegGatedCRF()
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = 5
    
    cls_loss = BCEWithLogitsLoss()
    contrastive_loss = SupConLoss(temperature=args.temp)
    
    # Initialize metrics
    train_accuracy = MultilabelAccuracy(num_labels=num_classes-1).cuda()
    avg_meter = pyutils.AverageMeter('loss', 'loss_ce', 'loss_pseudo', 'loss_crf', 'loss_cls')
    
    # Training setup
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    model.train()
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_epoch = 0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        train_metric_list = []
        
        for i_batch, sampled_batch in enumerate(trainloader):
            img, label = sampled_batch['image'], sampled_batch['label']
            img, label = img.cuda(), label.cuda()
            category = sampled_batch['category'].cuda()
            
            # Forward pass
            pred1, pred2, cls_output = model(img, ep=epoch_num, model_type="train") \
                if not args.no_class_rep else model(img, 0)
            
            outputs_soft1 = torch.softmax(pred1, dim=1)
            outputs_soft2 = torch.softmax(pred2, dim=1)
            
            # Calculate losses
            loss_ce1 = ce_loss(pred1, label[:].long())
            loss_ce2 = ce_loss(pred2, label[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2) if (label.unique() != 4).sum() else torch.tensor(0)
            loss = loss_ce
            
            # Pseudo supervision loss
            beta = random.random() + 1e-10
            if args.weight_pseudo_loss:
                pseudo_supervision = torch.argmax(
                    ((torch.min(outputs_soft1.detach(), outputs_soft2.detach()) > 0.5) * \
                     (beta * outputs_soft1.detach() + (1.0 - beta) * outputs_soft2.detach())),
                    dim=1, keepdim=False
                )
                loss_pse_sup = 0.5 * (
                    dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1)) +
                    dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1))
                )
                loss = loss + args.weight_pseudo_loss * loss_pse_sup
            
            ensemble_pred = (beta * outputs_soft1 + (1.0 - beta) * outputs_soft2)
            
            # CRF loss
            if args.weight_crf:
                out_gatedcrf = gatecrf_loss(
                    ensemble_pred,
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    img,
                    args.patch_size[0],
                    args.patch_size[1],
                )["loss"]
                loss = loss + args.weight_crf * out_gatedcrf
            
            # Classification loss
            if args.weight_cls:
                loss_cls = sum([cls_loss(o, category.float()) / len(cls_output) for o in cls_output])
                loss = loss + args.weight_cls * loss_cls
                preds = 0.5 * cls_output[0] + 0.5 * cls_output[1]
                acc = train_accuracy(preds, category)
            
            # Calculate training metrics
            if (epoch_num + 1) % args.val_every_epoches == 0:
                out = torch.argmax(ensemble_pred.detach(), dim=1)
                prediction = out.cpu().detach().numpy()
                metric_i = []
                for i in range(1, num_classes):
                    metric_i.append(calculate_metric_percase(
                        prediction == i, 
                        sampled_batch['gt'].cpu().detach().numpy() == i
                    ))
                train_metric_list.append(metric_i)
            
            # Backward pass
            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Learning rate decay
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item() if loss != 0 else 0, iter_num)
            
            # Update average meter
            avg_meter.add({
                'loss': loss.item() if loss != 0 else 0,
                'loss_crf': out_gatedcrf.item() if args.weight_crf != 0 else 0,
                'loss_cls': loss_cls.item() if args.weight_cls != 0 else 0
            })
            if loss_ce != 0:
                avg_meter.add({'loss_ce': loss_ce.item()})
            if args.weight_pseudo_loss != 0:
                avg_meter.add({'loss_pseudo': loss_pse_sup.item()})
            
            # Logging
            if iter_num % 200 == 0:
                logging.info(
                    'iteration %d : loss : %f, loss_ce: %f' %
                    (iter_num, loss.item() if loss != 0 else 0, loss_ce.item() if loss_ce != 0 else 0)
                )
            
            # Validation
            if iter_num > 1 and iter_num % 200 == 0:
                model.eval()
                
                # Training metrics
                if not args.val_mode and len(train_metric_list) > 0:
                    train_metric_list = np.nanmean(np.array(train_metric_list), axis=0)
                    train_dice = np.mean(train_metric_list, axis=0)[0]
                    mean_hd95 = np.mean(train_metric_list, axis=0)[1]
                    
                    for class_i in range(num_classes - 1):
                        writer.add_scalar('train/{}_dice'.format(class_i + 1),
                                        train_metric_list[class_i, 0], iter_num)
                        writer.add_scalar('train/{}_hd95'.format(class_i + 1),
                                        train_metric_list[class_i, 1], iter_num)
                
                # Validation metrics
                metric_list = []
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, 
                        classes=num_classes, patch_size=args.patch_size, epoch=epoch_num,
                        model_type='val' if not args.no_class_rep else None
                    )
                    metric_list.append(metric_i)
            
                metric_list = np.nanmean(np.array(metric_list), axis=0)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                    metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                    metric_list[class_i, 1], iter_num)
                
                performance = np.nanmean(metric_list, axis=0)[0]
                mean_hd95 = np.nanmean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                
                # Save best model
                if performance > best_performance:
                    best_performance = performance
                    best_epoch = epoch_num
                    save_mode_path = os.path.join(snapshot_path,
                                                'iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                           '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                
                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95)
                )
                model.train()
                train_metric_list = []
            
            # Save periodic checkpoints
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            
            if iter_num >= max_iterations:
                break
        
        avg_meter.pop()
        if iter_num >= max_iterations:
            iterator.close()
            break
    
    # Save final model
    final_model_path = os.path.join(snapshot_path, '{}_final_model.pth'.format(args.model))
    torch.save(model.state_dict(), final_model_path)
    
    logging.info('Best model at iteration %d with mean_dice: %.4f' % (best_epoch, best_performance))
    logging.info('Final model saved to {}'.format(final_model_path))
    
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

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)