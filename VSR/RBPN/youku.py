# import sys
# pycharm 不需要这操作
# sys.path.append("../../")
from utils.y4m_tools import read_y4m

import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
from PIL import Image, ImageOps
from .pyflow import pyflow

_TRANSFORM = Compose([ToTensor(), ])


def get_training_set(path_prefix, upscale_factor, data_augmentation, patch_size, future_frame):
    return YoukuDataset(path_prefix, upscale_factor, data_augmentation,
                        patch_size, future_frame, _TRANSFORM)


def get_eval_set(path_prefix, upscale_factor, data_augmentation, patch_size, future_frame):
    return YoukuDataset(path_prefix, upscale_factor, data_augmentation,
                        patch_size, future_frame, _TRANSFORM)


# def get_test_set(path_prefix, upscale_factor, future_frame):
#     return YoukuDatasetTest(path_prefix, upscale_factor, future_frame, _TRANSFORM)


class YoukuDataset(data.Dataset):
    def __init__(self, path_prefix, upscale_factor, augmentation,
                 patch_size, future_frame, transform=None):
        super(YoukuDataset, self).__init__()
        self.path_prefix = path_prefix  # such as dataset/Youku_00000
        self.upscale_factor = upscale_factor
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.future_frame = future_frame
        self.transform = transform
        self.l_frames, self.l_meta = read_y4m(path_prefix + "_l.y4m")
        self.gt_frames, self.gt_meta = read_y4m(path_prefix + "_h_GT.y4m")
        self.nFrames = 7
        return

    def __getitem__(self, index):
        lr, gt, neighbor = self._get_frame(index, self.future_frame)

        if self.patch_size != 0:
            lr, gt, neighbor, _ = get_patch(lr, gt, neighbor, self.patch_size,
                                            self.upscale_factor, self.nFrames)

        if self.augmentation:
            lr, gt, neighbor, _ = augment(lr, gt, neighbor)

        flow = [get_flow(lr, j) for j in neighbor]

        bicubic = rescale_img(lr, self.upscale_factor)

        if self.transform:
            gt = self.transform(gt)
            lr = self.transform(lr)
            bicubic = self.transform(bicubic)
            neighbor = [self.transform(j) for j in neighbor]
            # pylint: disable=E1101
            # RGB是 先行列，里面三通道三元组
            # transpose((2, 0, 1)) 后 先分三通道，然后再是行列
            flow = [torch.from_numpy(j.transpose((2, 0, 1))) for j in flow]
            # pylint: enable=E1101

        return lr, gt, neighbor, flow, bicubic

    def __len__(self):
        return self.nFrames

    def __add__(self, other):  # todo
        return data.dataset.ConcatDataset([self, other])

    def _get_frame(self, index, future):  # todo  正式训练时要用YUV的，不convert
        gt = Image.fromarray(self.gt_frames[index], mode='YCbCr').convert('RGB')
        lr = Image.fromarray(self.l_frames[index], mode='YCbCr').convert('RGB')
        if future:
            tt = int(self.nFrames / 2)
            seq = [x for x in range(4 - tt, 5 + tt) if x != 4]
        else:
            seq = [i for i in range(self.nFrames)]
            seq.reverse()
        neighbor = [Image.fromarray(self.l_frames[i]) for i in seq]
        return lr, gt, neighbor


# class YoukuDatasetTest(data.dataset):
#     def __init__(self, path_prefix, upscale_factor, future_frame, transform=None):
#         super(YoukuDatasetTest, self).__init__()
#         self.path_prefix = path_prefix  # such as data/Youku_00000
#         self.upscale_factor = upscale_factor
#         self.future_frame = future_frame
#         self.transform = transform
#         self.l_frames, self.l_meta = read_y4m(path_prefix + "_l.y4m")
#         self.frame_num = len(self.l_frames)
#         return
#
#     def __getitem__(self, index):
#         lr, gt, neighbor = self._get_frame(index, self.future_frame)
#
#         flow = [get_flow(lr, j) for j in neighbor]
#
#         bicubic = rescale_img(lr, self.upscale_factor)
#
#         if self.transform:
#             gt = self.transform(gt)
#             lr = self.transform(lr)
#             bicubic = self.transform(bicubic)
#             neighbor = [self.transform(j) for j in neighbor]
#             # pylint: disable=E1101
#             flow = [torch.from_numpy(j.transpose((2, 0, 1))) for j in flow]
#             # pylint: enable=E1101
#
#         return lr, gt, neighbor, flow, bicubic
#
#     def __len__(self):
#         return self.frame_num
#
#     def __add__(self, other):  # todo
#         return data.dataset.ConcatDataset([self, other])


class Error(Exception):
    pass


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    min_width = 20
    n_outer_fp_iterations = 7
    n_inner_fp_iterations = 1
    n_sor_iterations = 30
    col_type = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2w = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, min_width,
                                         n_outer_fp_iterations,
                                         n_inner_fp_iterations,
                                         n_sor_iterations, col_type)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # flow = rescale_flow(flow,0,1)
    return flow


def rescale_flow(x, max_range, min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range - min_range) / (max_val - min_val) * (x - max_val) + max_range


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def mod_crop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_in, img_tar, img_nn, patch_size, scale, n_frames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch


def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug
