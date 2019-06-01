import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps


class YoukuDataset(data.Dataset):
    def __init__(self, data_dir, upscale_factor, nFrames, augmentation, patch_size, padding):
        super(YoukuDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.nFrames = nFrames
        self.padding = padding
        self.paths = [v for v in glob.glob(f"{data_dir}/*/*_l_*.npy")]
        self.imgs = [os.path.basename(v) for v in self.paths]
        # self.__getitem__(0)
        # todo 抽帧 乱序 增强
        return

    def __getitem__(self, index):
        ref = self.imgs[index]
        gt_path = f"{self.data_dir}/{ref[:13]}/{ref}".replace('_l', '_h_GT')
        hr = read_npy(gt_path)
        files = self.generate_names(ref, self.padding)
        imgs = [read_npy(f"{self.data_dir}/{f[:13]}/{f}") for f in files]

        if self.augmentation:
            imgs, hr, _ = augment(imgs, hr)

        lr_seq = np.stack(imgs, axis=0)
        lr_seq = np.pad(lr_seq, ((0, 0), (1, 1), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
        lr_seq = torch.from_numpy(np.ascontiguousarray(lr_seq.transpose((0, 3, 1, 2)))).float()
        gt = np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))
        gt = torch.from_numpy(np.pad(gt, ((0, 0), (4, 4), (0, 0)), 'constant', constant_values=(0, 0))).float()
        return lr_seq.unsqueeze(0), gt

    def __len__(self):
        return len(self.imgs)

    def __add__(self, other):
        return data.dataset.ConcatDataset([self, other])

    def generate_names(self, file_name, padding='reflection'):
        """
        padding: replicate | reflection | new_info | circle
        :param file_name: 文件名
        :param padding: 补齐模式
        :return: 索引序列 [Youku_00000_l_100_00_.npy, ...]
        """
        fnl = file_name.split('_')
        max_n, crt_i = fnl[-3:-1]  # crt_i: 当前帧序号   max_n: 视频帧数
        id_len = len(crt_i)
        max_n, crt_i = int(max_n), int(crt_i)
        max_n = max_n - 1
        n_pad = self.nFrames // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < 0:
                if padding == 'replicate':
                    add_idx = 0
                elif padding == 'reflection':
                    add_idx = -i
                elif padding == 'new_info':
                    add_idx = (crt_i + n_pad) + (-i)
                elif padding == 'circle':
                    add_idx = self.nFrames + i
                else:
                    raise ValueError('Wrong padding mode')
            elif i >= max_n:
                if padding == 'replicate':
                    add_idx = max_n
                elif padding == 'reflection':
                    add_idx = max_n * 2 - i
                elif padding == 'new_info':
                    add_idx = (crt_i - n_pad) - (i - max_n)
                elif padding == 'circle':
                    add_idx = i - self.nFrames
                else:
                    raise ValueError('Wrong padding mode')
            else:
                add_idx = i
            fnl[-2] = str(add_idx).zfill(id_len)
            return_l.append('_'.join(fnl))
        return return_l


class Error(Exception):
    pass


def read_npy(path):
    return np.load(path).astype(np.float32) / 255


def augment(lr_seq, hr, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        hr = cv2.flip(hr, 1)
        lr_seq = [cv2.flip(lr, 1) for lr in lr_seq]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            hr = cv2.flip(hr, 0)
            lr_seq = [cv2.flip(lr, 0) for lr in lr_seq]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            hr = rotate(hr, 180)
            lr_seq = [rotate(lr, 180) for lr in lr_seq]
            info_aug['trans'] = True

    return lr_seq, hr, info_aug


def rotate(image, angle, center=None, scale=1.0):  # 1
    (h, w) = image.shape[:2]  # 2
    if center is None:  # 3
        center = (w // 2, h // 2)  # 4
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 5
    rotated = cv2.warpAffine(image, M, (w, h))  # 6
    return rotated  # 7
