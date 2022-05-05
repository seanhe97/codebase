"""
Paper: Crafting Better Contrastive Views for Siamese Representation Learning (CVPR 2022 Oral)
Arxiv: 2202.03278 
Code: github.com/xyupeng/ContrastiveCrop
简介： 提出了一种基于注意力的Crop方法，保证Crop不会把主体舍弃的同时增大多样性
"""

import random
import time
import math

from PIL import ImageFilter

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as F
from torchvision import datasets
from torch.distributions.beta import Beta
from torchvision.transforms import RandomResizedCrop, Compose



# /datasets/transforms/misc.py
class CCompose(Compose):
    r""" Apply transforms on bbox and img
    """
    def __call__(self, x):  # x: [sample, box]
        img = self.transforms[0](*x)
        for t in self.transforms[1:]:
            img = t(img)
        return img

# /datasets/transforms/misc.py
class MultiViewTransform:
    """Create multiple views of the same image"""
    def __init__(self, transform, num_views=2):
        if not isinstance(transform, (list, tuple)):
            transform = [transform for _ in range(num_views)]
        self.transforms = transform

    def __call__(self, x):
        views = [t(x) for t in self.transforms]
        return views

# /datasets/transforms/misc.py
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# DDP_simclr_ccrop.py
def update_box(eval_train_loader, model, len_ds, logger, t=0.05):
    r"""
    Args: 
        eval_train_loader
        model
        len_ds (int) : len(train_set)
    """
    if logger:
        logger.info(f'==> Start updating boxes...')
    model.eval()
    boxes = []
    t1 = time.time()
    for cur_iter, (images, _) in enumerate(eval_train_loader):  # drop_last=False
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feat_map = model(images, return_feat=True)  # (N, C, H, W)
        N, Cf, Hf, Wf = feat_map.shape
        eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
        eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
        eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0] # scale to [0, 1]
        eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
        eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')  # (N, 1, Hi, Wi)
        Hi, Wi = images.shape[-2:]

        for hmap in eval_train_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)

            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            # make sure bbox contains all salient region
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    gather_boxes = [torch.zeros_like(boxes) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_boxes, boxes)
    all_boxes = torch.stack(gather_boxes, dim=1).view(-1, 4)
    all_boxes = all_boxes[:len_ds]
    if logger is not None:  # cfg.rank == 0
        t2 = time.time()
        epoch_time = t2 - t1
        logger.info(f'Update box: {epoch_time}')
    return all_boxes

# datasets/transforms/ContrastiveCrop.py
class ContrastiveCrop(RandomResizedCrop):
    r""" ContrastiveCrop
    Args:
        alpha (float) : beta para
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        # a == b == 1.0 is uniform distribution
        self.beta = Beta(alpha, alpha)

    def get_params(self, img, box, scale, ratio):
        r"""Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = box
                ch0 = max(int(height * h0) - h//2, 0)
                ch1 = min(int(height * h1) - h//2, height - h)
                cw0 = max(int(width * w0) - w//2, 0)
                cw1 = min(int(width * w1) - w//2, width - w)

                i = ch0 + int((ch1 - ch0) * self.beta.sample())
                j = cw0 + int((cw1 - cw0) * self.beta.sample())
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, box):
        r"""
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, box, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)



# datasets/imagenet.py
class ImageFolderCCrop(datasets.ImageFolder):
    r""" Modified Dataset
    Args:
        root (str) : 
        transform_rcrop () :
        transform_ccrop () :
    """
    def __init__(self, root, transform_rcrop, transform_ccrop, init_box=[0., 0., 1., 1.], **kwargs):
        super().__init__(root=root, **kwargs)
        # transform
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop

        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.use_box:
            box = self.boxes[index].float().tolist()  # box=[h_min, w_min, h_max, w_max]
            sample = self.transform_ccrop([sample, box])
        else:
            sample = self.transform_rcrop(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


# models/resnet.py
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, zero_init_residual=False, maxpool=True):
        super().__init__()
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.maxpl = maxpool
        self.inplanes = 64
        if maxpool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpl:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # Layer4 Feature (N, C, H, W)
        if return_feat:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x