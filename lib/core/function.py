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

import numpy as np


def train_one_epoch(config, train_loader, model, loss_func, optimizer, epoch,
                    output_dir, tb_log_dir, writer_dict, scaler=None):
    logging.info('=> switch to train mode')
    model.train()
    total_loss = 0.

    for i, (gt) in enumerate(train_loader):
        optimizer.zero_grad()

        gt = [i.cuda() if isinstance(i, torch.Tensor) else i for i in gt]

        pred = model(gt[0])
        # print(np.array(pred.cpu()).shape)
        losses = loss_func(pred, gt)
        loss = sum(losses)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f}'.format(
        #                 epoch, i, loss, total_loss/(i + 1)))

        # measure elapsed time

        if i % config.PRINT_FREQ == 0:
            print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch, i, loss, total_loss/(i + 1)))
