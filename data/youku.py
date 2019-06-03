import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data


class YoukuDataset(data.Dataset):
    def __init__(self, data_dir, upscale_factor, nFrames, augmentation, patch_size, padding, v_freq=1, cut=False):
        super(YoukuDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.augmentation = augmentation
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.nFrames = nFrames
        self.padding = padding
        if cut:
            self.paths = [os.path.normpath(v) for v in glob.glob(f"{data_dir}/*_l*") if os.path.isdir(v)] * v_freq
        else:
            self.paths = [os.path.normpath(v) for v in glob.glob(f"{data_dir}/*_l")] * v_freq
        return

    def __getitem__(self, index):
        frame_paths = glob.glob(f"{self.paths[index]}\\*.npy")
        # 随机抽帧
        if len(frame_paths) >= self.nFrames:
            ref_id = random.randint(3, len(frame_paths) - 4)
        else:  # 有的太短
            ref_id = random.randint(0, len(frame_paths))
        # 取数据
        ref_path = frame_paths[ref_id]
        gt_path = f"{ref_path}".replace('_l', '_h_GT')  # 取GT
        hr = read_npy(gt_path)
        files = self.generate_names(ref_path, self.padding)
        imgs = [read_npy(f) for f in files]  # lr 序列

        if self.augmentation:
            imgs, hr, _ = augment(imgs, hr)

        if self.patch_size == 0:  # 不patch，要把图像补齐到4的倍数
            imgs = np.stack(imgs, axis=0)
            pad_size = (np.ceil(np.array(imgs.shape)[1:3] / 4) * 4 - np.array(imgs.shape)[1:3]).astype(np.int)
            imgs = np.pad(imgs, ((0, 0), (pad_size[0], pad_size[1]), (0, 0), (0, 0)), 'constant',
                          constant_values=(0, 0))
            hr = np.pad(hr, ((0, 0), (pad_size[0] * 4, pad_size[1] * 4), (0, 0)), 'constant',
                        constant_values=(0, 0))
            if hr.shape[-1] == 2048:
                hr = hr[:, 32:1120, 64:1984]
                imgs = imgs[:, :, 8:280, 16:496]
        else:  # patch
            imgs, hr = get_patch(imgs, hr, self.patch_size)
            imgs = np.stack(imgs, axis=0)
        # to tensor
        lr_seq = torch.from_numpy(np.ascontiguousarray(imgs.transpose((0, 3, 1, 2)))).float()
        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))).float()
        return lr_seq, gt

    def collate_fn(self, batch):
        lr_seq, gt = list(zip(*batch))
        return torch.stack(lr_seq), torch.stack(gt)

    def __len__(self):
        return len(self.paths)

    def __add__(self, other):
        return data.dataset.ConcatDataset([self, other])

    def generate_names(self, file_name, padding='reflection'):
        """
        padding: replicate | reflection | new_info | circle
        :param file_name: 文件名
        :param padding: 补齐模式
        :return: 索引序列 [Youku_00000_l-00_100_00_.npy, ...]
        """
        fnl = file_name.split('_')
        max_n, crt_i = fnl[-3:-1]  # crt_i: 当前帧序号   max_n: 视频帧数
        id_len = len(crt_i)
        max_n, crt_i = int(max_n), int(crt_i)
        max_n = max_n - 1
        n_pad = self.nFrames // 2
        return_l = []

        if max_n < self.nFrames:
            padding = 'replicate'

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
    rm = cv2.getRotationMatrix2D(center, angle, scale)  # 5
    rotated = cv2.warpAffine(image, rm, (w, h))  # 6
    return rotated  # 7


def get_patch(lr_seq, hr, patch_size):
    (h, w, _) = lr_seq[0].shape
    if patch_size > w or patch_size > h:
        raise ValueError('图像比patch小')
    elif patch_size % 4 != 0:
        raise ValueError('patch size 不是4的倍数')

    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    lr_seq = [lr[y:y + patch_size, x:x + patch_size, :] for lr in lr_seq]
    hr = hr[y << 2:(y + patch_size) << 2, x << 2:(x + patch_size) << 2, :]
    return lr_seq, hr
