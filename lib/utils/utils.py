from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import timedelta
from pathlib import Path

import os
import logging
import shutil
import time

import tensorwatch as tw
import torch
import torch.backends.cudnn as cudnn

from utils.comm import comm
from ptflops import get_model_complexity_info

import cv2
import numpy as np
import skimage
from einops import rearrange
# from dataset.COCOdataloader import UnNormalizer
from dataset.VOCdataloader import UnNormalizer

def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.txt'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + comm.head + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {} ...'.format(root_output_dir))
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print('=> setup logger ...')
    setup_logger(final_output_dir, cfg.RANK, phase)

    return str(final_output_dir)


def init_distributed(args):
    args.num_gpus = int(os.environ["WORLD_SIZE"]) \
        if "WORLD_SIZE" in os.environ else 1
    args.distributed = args.num_gpus > 1

    if args.distributed:
        print("=> init process group start")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(minutes=180))
        comm.local_rank = args.local_rank
        print("=> init process group end")


def setup_cudnn(config):
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def summary_model_on_master(model, config, output_dir, copy):
    if comm.is_main_process():
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../models', config.MODEL.NAME + '.py'),
            output_dir
        )
        logging.info('=> {}'.format(model))
        try:
            num_params = count_parameters(model)
            logging.info("Trainable Model Total Parameter: \t%2.1fM" % num_params)
        except Exception:
            logging.error('=> error when counting parameters')

        if config.MODEL_SUMMARY:
            try:
                logging.info('== model_stats by tensorwatch ==')
                df = tw.model_stats(
                    model,
                    (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
                )
                df.to_html(os.path.join(output_dir, 'model_summary.html'))
                df.to_csv(os.path.join(output_dir, 'model_summary.csv'))
                msg = '*'*20 + ' Model summary ' + '*'*20
                logging.info(
                    '\n{msg}\n{summary}\n{msg}'.format(
                        msg=msg, summary=df.iloc[-1]
                    )
                )
                logging.info('== model_stats by tensorwatch ==')
            except Exception:
                logging.error('=> error when run model_stats')

            try:
                logging.info('== get_model_complexity_info by ptflops ==')
                macs, params = get_model_complexity_info(
                    model,
                    (3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0]),
                    as_strings=True, print_per_layer_stat=True, verbose=True
                )
                logging.info(f'=> FLOPs: {macs:<8}, params: {params:<8}')
                logging.info('== get_model_complexity_info by ptflops ==')
            except Exception:
                logging.error('=> error when run get_model_complexity_info')


def resume_checkpoint(model,
                      optimizer,
                      config,
                      output_dir,
                      in_epoch):
    best_perf = 0.0
    begin_epoch_or_step = 0

    checkpoint = os.path.join(output_dir, 'checkpoint.pth')\
        if not config.TRAIN.CHECKPOINT else config.TRAIN.CHECKPOINT

    if config.TRAIN.AUTO_RESUME and os.path.exists(checkpoint):
        logging.info(
            "=> loading checkpoint '{}'".format(checkpoint)
        )
        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        # best_perf = checkpoint_dict['perf']
        # begin_epoch_or_step = checkpoint_dict['epoch' if in_epoch else 'step']
        state_dict = checkpoint_dict['state_dict']
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        logging.info(
            "=> {}: loaded checkpoint '{}' ({}: {})"
            .format(comm.head,
                    checkpoint,
                    'epoch' if in_epoch else 'step',
                    begin_epoch_or_step)
        )

    return best_perf, begin_epoch_or_step


def save_checkpoint_on_master(model,
                              *,
                              distributed,
                              model_name,
                              optimizer,
                              output_dir,
                              in_epoch,
                              epoch_or_step,
                              best_perf):
    if not comm.is_main_process():
        return

    states = model.module.state_dict() \
        if distributed else model.state_dict()

    logging.info('=> saving checkpoint to {}'.format(output_dir))
    save_dict = {
        'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'model': model_name,
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }

    try:
        torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))
    except Exception:
        logging.error('=> error when saving checkpoint!')


def save_model_on_master(model, distributed, out_dir, fname):
    if not comm.is_main_process():
        return

    try:
        fname_full = os.path.join(out_dir, fname)
        logging.info(f'=> save model to {fname_full}')
        torch.save(
            model.module.state_dict() if distributed else model.state_dict(),
            fname_full
        )
    except Exception:
        logging.error('=> error when saving checkpoint!')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    from collections import OrderedDict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def visualize_image(data, model, epoch, labels):
    saved_path = 'visualize'
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    unnormalize = UnNormalizer()

    with torch.no_grad():
        # print(data[0].shape)
        # x = data['img']
        x = data[0]
        # x = rearrange(x, 'h w c -> c h w')
        x = x.unsqueeze(0).float()
        x = x.cuda()

        scores, classification, transformed_anchors = model(x)
        idxs = np.where(scores.cpu()>torch.mean(scores.cpu()))
        x = x.cpu()

        # new_data = unfold(data)
        # x = new_data['img']
        # img = np.array(255 * unnormalize(x[:, :, :])).copy()

        img = np.array(255 * unnormalize(x[0, :, :, :])).copy()
        img[img<0] = 0
        img[img>255] = 255
        img = np.transpose(img, (1, 2, 0))
        
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # if epoch == 0:
        #     cv2.imwrite(saved_path + '/original.jpg', img)
        #     # y = new_data['annot'].numpy()
        #     y = data['annot'].numpy()

        #     for each_label in y:
        #         bbox = each_label[:4]
        #         x1 = int(bbox[0])
        #         y1 = int(bbox[1])
        #         x2 = int(bbox[2])
        #         y2 = int(bbox[3])
        #         label_name = labels[int(each_label[4])]
        #         draw_caption(img, (x1, y1, x2, y2), label_name)
        #         cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)

        #     cv2.imwrite(saved_path + '/annotation.jpg', img)


        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = labels[int(classification[idxs[0][j]])]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)

        cv2.imwrite(saved_path + '/result_{}.jpg'.format(epoch), img)

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def unfold(data):
    x = data['img']
    y = data['annot']
    w_scale = data['w_scale']
    h_scale = data['h_scale']

    x = data['img']
    x = rearrange(x, 'h w c -> c h w')
    x = x.float()

    cur_size = 224
    w_size = int(round(cur_size * w_scale, 0))
    h_size = int(round(cur_size * h_scale, 0))


    image = x.numpy()
    image = np.transpose(image, (1, 2, 0))
    y = y.numpy()

    y[:, 0] *= h_scale
    y[:, 1] *= w_scale
    y[:, 2] *= h_scale
    y[:, 3] *= w_scale

    # new_image = skimage.transform.resize(image, (w_size, h_size))
    new_image = cv2.resize(image, (w_size, h_size))

    return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(y)}