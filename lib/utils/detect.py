from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pprint import pprint
import os

def preprocess_img(img, input_ksize):
    min_side, max_side = input_ksize
    h, w = img.height, img.width
    _pad = 32
    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    img_resized = np.array(img.resize((nw, nh)))

    pad_w = _pad - nw % _pad
    pad_h = _pad - nh % _pad

    img_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    img_paded[:nh, :nw, :] = img_resized

    return img_paded, {'raw_height': h, 'raw_width': w}


def show_img(img, boxes, clses, scores, epoch):
    saved_path = 'visualize'
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    boxes, scores = [i.cpu() for i in [boxes, scores]]

    boxes = boxes.long()
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(xy=box.tolist(), outline='red', width=3)

    boxes = boxes.tolist()
    scores = scores.tolist()
    plt.figure(figsize=(10, 10))
    for i in range(len(boxes)):
        plt.text(x=boxes[i][0], y=boxes[i][1], s='{}: {:.4f}'.format(clses[i], scores[i]), wrap=True, size=15,
                 bbox=dict(facecolor="r", alpha=0.7))
    # plt.imshow(img)
    # plt.show()
    img.save(saved_path + '/result_{}.jpg'.format(epoch))

@torch.no_grad()
def detect(model, epoch, labels):
    img = Image.open('./original.jpg').convert('RGB')
    img_paded, info = preprocess_img(img, [224, 224])

    imgs = [img]
    infos = [info]

    input = transforms.ToTensor()(img_paded)
    mean = [0.40789654, 0.44719302, 0.47026115]
    std = [0.28863828, 0.27408164, 0.27809835]
    input = transforms.Normalize(std=std, mean=mean)(input)
    inputs = input.unsqueeze(0).cuda()
    print('preprocess done!\ninit model...')

    model = model.eval()
    detects = model.inference(inputs, topK=40, return_hm=False, th=0.25, CLASSES_NAME=labels)

    for img_idx in range(len(detects)):
        boxes = detects[img_idx][0]
        scores = detects[img_idx][1]
        clses = detects[img_idx][2]

        img = imgs[img_idx]
        show_img(img, boxes, clses, scores, epoch)

