import argparse
import logging
import os
import random
import shutil
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

from Scribble_DualTeachers.code.utils.pick_reliable_pixels import process_pseudo_label
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
from val import test_single_volume
from utils.pick_reliable_pixels import refine_high_confidence
from utils.ema_optim import WeightEMA

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/MSCMR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='DualTeacher', help='experiment_name')
parser.add_argument('--data', type=str,
                    default='MSCMR', help='experiment_name')
parser.add_argument('--sup_type', type=str,
                    default='scribble', help='supervision')
parser.add_argument('--model', type=str,
                    default='unet_hl', help='name of network')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel')
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
parser.add_argument('--consistency_rampup', type=float, default=40, help='use automatic mixed precision')
parser.add_argument('--confidence_threshold', type=float, default=0.5, help='experiment_name')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1 * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    # Network definition
    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    # ===========
    # Models
    # =========== 
    model = create_model(ema=False) # student
    teacher1 = create_model(ema=True) # teacher1
    teacher2 = create_model(ema=True) # teacher2
    model.cuda()
    teacher1.cuda()
    teacher2.cuda()
    model.train()
    teacher1.train()
    teacher2.train()

    # ===========
    # Datasets
    # =========== 
    db_train = MSCMRDataSets(base_dir=args.root_path, split="train", 
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]),
                            sup_type=args.sup_type)
    db_val = MSCMRDataSets(base_dir=args.root_path, split="val")

    # ===========
    # DataLoaders
    # =========== 
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # ===========
    # Optimizer+Loss
    # =========== 
    optimizer = optim.SGD(model.parameters(), lr=base_lr, 
                          momentum=0.9, weight_decay=0.0001)
    tea1_optimizer = WeightEMA(model, teacher1, 0.99)
    tea2_optimizer = WeightEMA(model, teacher2, 0.99)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)

    # ===========
    # Training
    # ===========    
     
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    alpha = 0.99

    for epoch_num in iterator:
        for i_batch, sampled in enumerate(trainloader):

            # Get data
            image = sampled['image'].cuda() 
            scrib = sampled['label'].cuda().long()


            # Forward pass
            with torch.no_grad():
                teacher1_output, high1, low1 = teacher1(image)
                outputs_soft_teacher1 = torch.softmax(teacher1_output, dim=1)
                teacher2_output, high2, low2 = teacher2(image)
                outputs_soft_teacher2 = torch.softmax(teacher2_output, dim=1)

            student_output, high, low = model(image)
            outputs_soft_student = torch.softmax(student_output, dim=1)

            loss_ce_stu = ce_loss(student_output, scrib[:])
            loss_ce_tea1 = ce_loss(teacher1_output, scrib[:])
            loss_ce_tea2 = ce_loss(teacher2_output, scrib[:])

            pseudo_label1 = refine_high_confidence(outputs_soft_teacher1, threshold=args.confidence_threshold)
            pseudo_label2 = refine_high_confidence(outputs_soft_teacher2, threshold=args.confidence_threshold)

            if loss_ce_tea1 < loss_ce_tea2:
                mode = 1
                loss_pseudo = ce_loss(student_output, pseudo_label1[:].long()) + dice_loss(outputs_soft_student, pseudo_label1.unsqueeze(1))
                loss_low = (F.l1_loss(low1, low) + (1 - F.cosine_similarity(low1.flatten(1), low.flatten(1)).mean())) / 2
                loss_high = (F.l1_loss(high1, high) + (1 - F.cosine_similarity(high1.flatten(1), high.flatten(1)).mean())) / 2

            else:
                mode = 0
                loss_pseudo = ce_loss(student_output, pseudo_label2[:].long()) + dice_loss(outputs_soft_student, pseudo_label2.unsqueeze(1))
                loss_low = (F.l1_loss(low2, low) + (1 - F.cosine_similarity(low2.flatten(1), low.flatten(1)).mean())) / 2
                loss_high = (F.l1_loss(high2, high) + (1 - F.cosine_similarity(high2.flatten(1), high.flatten(1)).mean())) / 2
                

            consistency_weight = get_current_consistency_weight(iter_num // 300, args)
            loss = loss_ce_stu + loss_pseudo * 0.5 * consistency_weight + (loss_low + loss_high) * 0.5 * consistency_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if mode == 1:
                tea1_optimizer.step()
            else:
                tea2_optimizer.step()


            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar("train/loss_total", loss.item(), iter_num)
            writer.add_scalar("train/loss_pseudo_label", loss_pseudo.item(), iter_num)
            writer.add_scalar("train/loss_ce", loss_ce_stu.item(), iter_num)
            writer.add_scalar("train/loss_low", loss_low.item(), iter_num)
            writer.add_scalar("train/loss_high", loss_high.item(), iter_num)

            # ===========
            # Evaluation
            # =========== 
            if iter_num % 400 == 0:
                logging.info(
                'iteration %d : loss : %f, loss_pseudo_label: %f' 
                % (iter_num, loss.item(), loss_pseudo.item()))

            if iter_num > 1 and iter_num % 400 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
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
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)