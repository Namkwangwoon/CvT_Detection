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


from dataset.COCOdataloader import CocoDataset, Normalizer, Augmenter, Resizer, CSVDataset, AspectRatioBasedSampler, collater
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.SOCdataloader import Config, get_loader
from eval_coco import evaluate_coco
from core.losses import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

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
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    model = build_model(config)
    # model.load_state_dict(torch.load('OUTPUT/imagenet/cvt-13-224x224/cvt_transformer_50.pth'))
    model.to(torch.device('cuda'))

    optimizer = build_optimizer(config, model)
    print(optimizer)

    dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, drop_last=False)
    train_loader = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=config.TEST.BATCH_SIZE_PER_GPU, drop_last=False)
    valid_loader = DataLoader(dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val)

    criterion = FocalLoss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[24, 80], gamma=0.1)

    for epoch in range(config.TRAIN.END_EPOCH):
        # train_one_epoch(config, train_loader, model, criterion, optimizer,
                        # epoch, final_output_dir, tb_log_dir, writer_dict,
                        # scaler=scaler)
        ### Train ###
        for iter_num, (x, y) in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                classification_loss, regression_loss = model([x, y])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                optimizer.step()

                if iter_num % config.PRINT_FREQ == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch, iter_num, float(classification_loss), float(regression_loss), float(loss)))

            except Exception as e:
                print(e)
                continue

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            try:
                evaluate_coco(dataset_val, model)
            except Exception as e:
                print(e)
                print()
        else:
            model.eval()

        fname = f'cvt_transformer_{epoch}.pth'
        fname_full = os.path.join(final_output_dir, fname)
        torch.save(
            model.module.state_dict() if args.distributed else model.state_dict(),
            fname_full
        )

        lr_scheduler.step(epoch=epoch+1)
        lr = lr_scheduler.get_last_lr()[0]
        print(f'=> lr : {lr}')

if __name__ == '__main__':
    main()
