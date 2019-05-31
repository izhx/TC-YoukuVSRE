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
        self.__getitem__(0)
        return

    def __getitem__(self, index):
        ref = self.imgs[index]
        gt_path = f"{self.data_dir}/{ref[:13]}/{ref}".replace('_l', '_h_GT')
        hr = read_npy(gt_path)
        files = self.generate_names(ref, self.padding)
        imgs = [read_npy(f"{self.data_dir}/{f[:13]}/{f}") for f in files]

        if self.patch_size != 0:
            imgs, hr, _ = get_patch(imgs, hr, self.patch_size, self.upscale_factor)

        if self.augmentation:
            imgs, hr, _ = augment(imgs, hr)

        lr_seq = np.stack(imgs, axis=0)  # todo gt待检验形状
        lr_seq = torch.from_numpy(np.ascontiguousarray(lr_seq.transpose((0, 3, 1, 2)))).float()
        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))).float()
        return lr_seq, gt

    def __len__(self):
        return len(self.imgs)

    def __add__(self, other):  # todo
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


def read_npy(path):
    return np.load(path).astype(np.float32) / 255


def read_image(img_path):
    """read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    """
    img_gt = cv2.imread(img_path)
    img = img_gt.astype(np.float32) / 255.
    return img


def read_seq_imgs(img_seq_path):
    """
    read a sequence of images
    :param img_seq_path:
    :return:
    """
    img_path_l = sorted(glob.glob(img_seq_path + '/*'))
    img_l = [read_image(v) for v in img_path_l]
    # stack to T C H W, RGB, [0,1], torch
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


class Error(Exception):
    pass


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


def get_patch(hr, lr_seq, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lr_seq[0].size
    tp = scale * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    hr = hr.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    lr_seq = [j.crop((iy, ix, iy + ip, ix + ip)) for j in lr_seq]  # [:, iy:iy + ip, ix:ix + ip]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return lr_seq, hr, info_patch


def augment(lr_seq, hr, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        hr = ImageOps.flip(hr)
        lr_seq = [ImageOps.flip(j) for j in lr_seq]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            hr = ImageOps.mirror(hr)
            lr_seq = [ImageOps.mirror(j) for j in lr_seq]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            hr = hr.rotate(180)
            lr_seq = [j.rotate(180) for j in lr_seq]
            info_aug['trans'] = True

    return hr, lr_seq, info_aug
