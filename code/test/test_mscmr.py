import argparse
import os
import shutil
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
np.bool = np.bool_
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../../data/MSCMR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='MSCMR/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_hl', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/MSCMR_testing_volumes/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_hl":
                out_main, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    return first_metric, second_metric, third_metric


def Inference(FLAGS):

    test_dir = os.path.join(FLAGS.root_path, "MSCMR_testing_volumes")
    image_list = [
        os.path.splitext(f)[0]
        for f in os.listdir(test_dir)
        if f.endswith(".h5")
    ]
    image_list.sort()
    save_mode_path = "../../checkpoints/MSCMR_SDTNet/unet_hl_best_model.pth"
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print("RV:", metric[0], " | Myo:", metric[1], " | LV:", metric[2])
    print((metric[0]+metric[1]+metric[2])/3)
