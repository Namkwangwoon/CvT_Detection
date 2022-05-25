import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image, ImageEnhance


class Config():
    def __init__(self) -> None:
        self.image_root = 'datasets/SOC/TrainSet/Imgs/'
        self.gt_root = 'datasets/SOC/TrainSet/gt/'

        # self-supervision
        self.lambda_loss_ss = 0.3   # 0 means no self-supervision

        # label smoothing
        self.label_smooth = 0.001   # epsilon for smoothing, 0 means no label smoothing, 

        # # preproc
        # self.preproc_activated = True
        # self.hflip_prob = 0.5
        # self.crop_border = 30      # < 1 as percent of min(wid, hei), >=1 as pixel
        # self.rotate_prob = 0.2
        # self.rotate_angle = 15
        # self.enhance_activated = True
        # self.enhance_brightness = (5, 15)
        # self.enhance_contrast = (5, 15)
        # self.enhance_color = (0, 20)
        # self.enhance_sharpness = (0, 30)
        # self.gaussian_mean = 0.1
        # self.gaussian_sigma = 0.35
        # self.pepper_noise = 0.0015
        # self.pepper_turn = 0.5

        
# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png') or f.endswith('.PNG')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.config = Config()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        # if self.config.preproc_activated:
        #     if self.config.hflip_prob:
        #         image, gt = cv_random_flip(image, gt, prob=self.config.hflip_prob)
        #     if self.config.crop_border:
        #         image, gt = randomCrop(image, gt, border=self.config.crop_border)
        #     if self.config.rotate_prob:
        #         image, gt = randomRotation(image, gt, prob=self.config.rotate_prob, angle=self.config.rotate_angle)
        #     if self.config.enhance_activated:
        #         image = colorEnhance(image, self.config.enhance_brightness, self.config.enhance_contrast, self.config.enhance_color, self.config.enhance_sharpness)
        #     if self.config.gaussian_sigma:
        #         gt = randomGaussian(gt)
        #     if self.config.pepper_noise and self.config.pepper_turn:
        #         gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('I') 

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SalObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# test dataset and loader
class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

