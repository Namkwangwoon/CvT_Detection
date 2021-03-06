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
from torch.utils.data import Sampler
from utils.utils import visualize_image

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

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='DATASET/coco')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    init_distributed(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    model = build_model(config)
    model.to(torch.device('cuda'))

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nhwc':
        logging.info('=> convert memory format to nhwc')
        model.to(memory_format=torch.channels_last)

    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_perf = 0.0
    best_model = True
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model)
    print(optimizer)

    ### COCO dataset ###
    dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, drop_last=False)
    # train_loader = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)


    # dataset_train = CocoDataset(args.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, drop_last=False)
    # train_sequential_sampler = Sampler.SequentialSampler(dataset_train, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, drop_last=True)
    # train_loader = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=train_sequential_sampler)
    train_loader = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, shuffle=False, drop_last=True)

    if dataset_val is not None:
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=config.TEST.BATCH_SIZE_PER_GPU, drop_last=False)
        # valid_loader = DataLoader(dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val)
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        # val_sequential_sampler = Sampler.SequentialSampler(dataset_val, batch_size=config.TEST.BATCH_SIZE_PER_GPU, drop_last=True)
        # valid_loader = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=val_sequential_sampler)
        valid_loader = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_size=config.TEST.BATCH_SIZE_PER_GPU, shuffle=False, drop_last=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    criterion = FocalLoss()

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 120], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> start training')
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        logging.info('=> {} train start'.format(head))
        # with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
        #     train_one_epoch(config, train_loader, model, criterion, optimizer,
        #                     epoch, final_output_dir, tb_log_dir, writer_dict,
        #                     scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on validation set
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        model.eval()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            # print(dataset_val.labels)
            try:
                visualize_image(dataset_val[0], model, epoch, dataset_val.labels)
                evaluate_coco(dataset_val, model)
            except Exception as e:
                print(e)
                print()

        perf=0
        best_model = (perf > best_perf)
        best_perf = perf if best_model else best_perf

        fname = f'cvt_transformer_{epoch}.pth'
        fname_full = os.path.join(final_output_dir, fname)
        torch.save(
            model.module.state_dict() if args.distributed else model.state_dict(),
            fname_full
        )

        lr_scheduler.step(epoch=epoch+1)
        lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    writer_dict['writer'].close()
    logging.info('=> finish training')

if __name__ == '__main__':
    main()
