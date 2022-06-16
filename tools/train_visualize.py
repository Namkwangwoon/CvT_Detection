
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter
import numpy as np
import cv2

import _init_paths
from config import config
from config import update_config
from config import save_config
from core.loss import build_criterion
from core.function import train_one_epoch
from dataset import build_dataloader
from models import build_model
from optim import build_optimizer
from scheduler import build_lr_scheduler
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import resume_checkpoint
from utils.utils import save_checkpoint_on_master
from utils.utils import save_model_on_master


from dataset.COCOdataloader import CocoDataset, Normalizer, UnNormalizer, Augmenter, Resizer, CSVDataset, AspectRatioBasedSampler, collater
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast
from dataset.SOCdataloader import Config, get_loader
from eval_coco import evaluate_coco

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Detection network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/imagenet/cvt/cvt-13-224x224.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='DATASET/coco')
    parser.add_argument('--model_path', help='Path to model checkpoint directory')

    args = parser.parse_args()

    return args

def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    model = build_model(config)
    model.load_state_dict(torch.load(args.model_path))
    # model = torch.load()
    model.training = False
    model.to(torch.device('cuda:0'))
    model.eval()

    ### COCO dataset ###
    # dataset_val = CocoDataset(args.coco_path, set_name='val2017',
    #                             transform=transforms.Compose([Normalizer(), Resizer()]))

    dataset_val = CocoDataset(args.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Resizer()]))

    if dataset_val is not None:
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=56, drop_last=False)
        # valid_loader = DataLoader(dataset_val, num_workers=8, collate_fn=collater, batch_sampler=sampler_val)
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        valid_loader = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # evaluate_coco(dataset_val, model)

    unnormalize = UnNormalizer()
    for idx, (x, y) in enumerate(valid_loader):
        print('============ X, Y ============')
        print(x.shape)
        print(y.shape)
        print()

        with torch.no_grad():
            img = np.array(255 * unnormalize(x[0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite('original.jpg', img)

            y = np.array(y)
            print(y)
            print(y.shape)

            for j in range(y.shape[1]):
                bbox = y[0][j][:4]
                cls_label = y[0][j][4]
                label_name = dataset_val.labels[int(cls_label)]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)

            cv2.imwrite('result.jpg', img)
        break


if __name__ == '__main__':
    main()