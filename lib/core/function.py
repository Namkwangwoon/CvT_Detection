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
        print('============ X, Y ============')
        print(x.shape)
        print(y.shape)
        for yy in y:
            print('============ YY ============')
            print(yy)
            print()
        print()

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