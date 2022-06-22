import torch
from torch import nn
import numpy as np

from lib.core.centernet.utils import *
from lib.core.centernet.losses import *

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.down_stride = 4

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = 1.
        self.beta = 0.1
        self.gamma = 1.

    def forward(self, pred, gt):
        pred_hm, pred_wh, pred_offset = pred
        imgs, gt_boxes, gt_classes, gt_hm, infos = gt
        gt_nonpad_mask = gt_classes.gt(0)

        # print('pred_hm: ', pred_hm.shape, '  gt_hm: ', gt_hm.shape)
        cls_loss = self.focal_loss(pred_hm, gt_hm)

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0
        for batch in range(imgs.size(0)):
            ct = infos[batch]['ct'].cuda()
            ct_int = ct.long()
            num += len(ct_int)
            batch_pos_pred_wh = pred_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]]
            wh = torch.stack([
                batch_boxes[:, 2] - batch_boxes[:, 0],
                batch_boxes[:, 3] - batch_boxes[:, 1]
            ]).view(-1) / self.down_stride
            offset = (ct - ct_int.float()).T.contiguous().view(-1)

            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (num + 1e-6)
