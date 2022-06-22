
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
    dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    # dataset_val = CocoDataset(args.coco_path, set_name='train2017',
                                    # transform=transforms.Compose([Normalizer(), Resizer()]))

    if dataset_val is not None:
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=56, drop_last=False)
        # valid_loader = DataLoader(dataset_val, num_workers=8, collate_fn=collater, batch_sampler=sampler_val)
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        valid_loader = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # evaluate_coco(dataset_val, model)

    unnormalize = UnNormalizer()
    for idx, (x, y) in enumerate(valid_loader):
        with torch.no_grad():
            st = time.time()
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            
            with autocast(enabled=config.AMP.ENABLED):
                if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nwhc':
                    x = x.contiguous(memory_format=torch.channels_last)
                    y = y.contiguous(memory_format=torch.channels_last)
            
                scores, classification, transformed_anchors = model(x)
            
            print('============ TRANSFORMED_ANCHORS ============')
            print(transformed_anchors.shape)
            print(transformed_anchors)
            print()
            
            # idxs = np.where(scores.cpu()>torch.mean(scores.cpu()))
            idxs = np.where(scores.cpu()>0)
            print('============ TRANSFORMED_ANCHORS ============')
            print(idxs)
            print('num final bbox : ', len(idxs[0]))
            print()
            
            print('Elapsed time: {}'.format(time.time()-st))
            print('============ SCORE ============')
            print(scores.shape)
            print(scores)
            # print(torch.max(scores))  
            print("mean score :", torch.mean(scores))
            print()

            print('============ CLASSIFICATION ============')
            print(classification.shape)
            print(classification)
            print()

            print('========== GT ==========')
            print(y)
            print()

            x = x.cpu()
            img = np.array(255 * unnormalize(x[0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            cv2.imwrite('original.jpg', img)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                # print('box : ', (x1, y1, x2, y2))

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
                # print(label_name)

            cv2.imwrite('result.jpg', img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
        break


if __name__ == '__main__':
    main()