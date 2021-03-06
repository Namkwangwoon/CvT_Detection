from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import torch

from timm.data import Mixup
from torch.cuda.amp import autocast

from core.evaluate import accuracy
from utils.comm import comm


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, scaler=None):
    logging.info('=> switch to train mode')
    model.train()

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        with autocast(enabled=config.AMP.ENABLED):
            if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nwhc':
                x = x.contiguous(memory_format=torch.channels_last)
                y = y.contiguous(memory_format=torch.channels_last)

            # outputs = model(x)
            # loss = criterion(outputs, y)

            classification_loss, regression_loss = model([x, y])
            # inputs =model(x)

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            # classification_output, regression_output = model(x)


        # compute gradient and do update step
        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') \
            and optimizer.is_second_order

        scaler.scale(loss).backward(create_graph=is_second_order)

        if config.TRAIN.CLIP_GRAD_NORM > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD_NORM
            )

        scaler.step(optimizer)
        scaler.update()

        # print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
        #                 epoch, i, float(classification_loss), float(regression_loss), loss))

        # measure elapsed time

        if i % config.PRINT_FREQ == 0:
            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch, i, float(classification_loss), float(regression_loss), loss))

        torch.cuda.synchronize()
        

@torch.no_grad()
def test(config, val_loader, model, criterion, output_dir, tb_log_dir,
         writer_dict=None, distributed=False, real_labels=None,
         valid_labels=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logging.info('=> switch to eval mode')
    model.eval()

    end = time.time()
    for i, (x, y) in enumerate(val_loader):
        # compute output
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        outputs = model(x)
        if valid_labels:
            outputs = outputs[:, valid_labels]

        loss = criterion(outputs, y)

        if real_labels and not distributed:
            real_labels.add_result(outputs)

        # measure accuracy and record loss
        losses.update(loss.item(), x.size(0))

        prec1, prec5 = accuracy(outputs, y, (1, 5))
        top1.update(prec1, x.size(0))
        top5.update(prec5, x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logging.info('=> synchronize...')
    comm.synchronize()
    top1_acc, top5_acc, loss_avg = map(
        _meter_reduce if distributed else lambda x: x.avg,
        [top1, top5, losses]
    )

    if real_labels and not distributed:
        real_top1 = real_labels.get_accuracy(k=1)
        real_top5 = real_labels.get_accuracy(k=5)
        msg = '=> TEST using Reassessed labels:\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                top1=real_top1,
                top5=real_top5,
                error1=100-real_top1,
                error5=100-real_top5
            )
        logging.info(msg)

    if comm.is_main_process():
        msg = '=> TEST:\t' \
            'Loss {loss_avg:.4f}\t' \
            'Error@1 {error1:.3f}%\t' \
            'Error@5 {error5:.3f}%\t' \
            'Accuracy@1 {top1:.3f}%\t' \
            'Accuracy@5 {top5:.3f}%\t'.format(
                loss_avg=loss_avg, top1=top1_acc,
                top5=top5_acc, error1=100-top1_acc,
                error5=100-top5_acc
            )
        logging.info(msg)

    if writer_dict and comm.is_main_process():
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', loss_avg, global_steps)
        writer.add_scalar('valid_top1', top1_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    logging.info('=> switch to train mode')
    model.train()

    return top1_acc


def _meter_reduce(meter):
    rank = comm.local_rank
    meter_sum = torch.FloatTensor([meter.sum]).cuda(rank)
    meter_count = torch.FloatTensor([meter.count]).cuda(rank)
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count