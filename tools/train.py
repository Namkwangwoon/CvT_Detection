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
from core.function import train_one_epoch, test
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
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
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

    # best_perf, begin_epoch = resume_checkpoint(
        # model, optimizer, config, final_output_dir, True
    # )

    # train_loader = build_dataloader(config, True, args.distributed)
    # valid_loader = build_dataloader(config, False, args.distributed)

    ### COCO dataset ###
    
    # Create the data loaders
    if args.dataset == 'coco':

        if args.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(args.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(args.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif args.dataset == 'csv':

        if args.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if args.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=args.csv_train, class_list=args.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if args.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=args.csv_val, class_list=args.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, drop_last=False)
    train_loader = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=config.TEST.BATCH_SIZE_PER_GPU, drop_last=False)
        valid_loader = DataLoader(dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val)
        
    ###
    
    ### SOC dataset ###
    
    # config = Config()

    # train_image_root = 'DATASET/SOC/TrainSet/Imgs/'
    # train_gt_root = 'DATASET/SOC/TrainSet/gt/'
    # valid_image_root = 'DATASET/SOC/ValSet/Imgs/'
    # valid_gt_root = 'DATASET/SOC/ValSet/gt/'
    # train_loader = get_loader(train_image_root, train_gt_root, batchsize=config.TRAIN.BATCH_SIZE_PER_GPU, trainsize=224)
    # valid_loader = get_loader(valid_image_root, valid_gt_root, batchsize=config.TEST.BATCH_SIZE_PER_GPU, trainsize=224)
    
    ###

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    criterion = build_criterion(config)
    criterion.cuda()
    # criterion_eval = build_criterion(config, train=False)
    # criterion_eval.cuda()

    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

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
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, final_output_dir, tb_log_dir, writer_dict,
                            scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on validation set
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            # perf = test(
            #     config, valid_loader, model, criterion_eval,
            #     final_output_dir, tb_log_dir, writer_dict,
            #     args.distributed
            # )

            evaluate_coco(dataset_val, model)
        
        else:
            model.eval()

        perf=0
        best_model = (perf > best_perf)
        best_perf = perf if best_model else best_perf

        fname = f'cvt_transformer_{epoch}.pth'
        fname_full = os.path.join(final_output_dir, fname)
        torch.save(
            model.module.state_dict() if distributed else model.state_dict(),
            fname_full
        )

        lr_scheduler.step(epoch=epoch+1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        # save_checkpoint_on_master(
        #     model=model,
        #     distributed=args.distributed,
        #     model_name=config.MODEL.NAME,
        #     optimizer=optimizer,
        #     output_dir=final_output_dir,
        #     in_epoch=True,
        #     epoch_or_step=epoch,
        #     best_perf=best_perf,
        # )

        # save_model_on_master(
        #     model, args.distributed, final_output_dir, f'model_{epoch}.pth'
        # )

        # if best_model and comm.is_main_process():
        #     save_model_on_master(
        #         model, args.distributed, final_output_dir, 'model_best.pth'
        #     )

        # if config.TRAIN.SAVE_ALL_MODELS and comm.is_main_process():
        #     save_model_on_master(
        #         model, args.distributed, final_output_dir, f'model_{epoch}.pth'
        #     )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    # save_model_on_master(
    #     model, args.distributed, final_output_dir, 'final_state.pth'
    # )

    # if config.SWA.ENABLED and comm.is_main_process():
    #     save_model_on_master(
    #          args.distributed, final_output_dir, 'swa_state.pth'
    #     )

    # torch.save(model, '{}_cvt_transformer_{}.pt'.format('coco', epoch_num))
    

    writer_dict['writer'].close()
    logging.info('=> finish training')


if __name__ == '__main__':
    main()
