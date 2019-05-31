import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps


# import pyflow


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
        files = self.generate_names(ref, self.padding)
        imgs = [read_npy(f"{self.data_dir}/{f[:13]}/{f}") for f in files]
        lr_seq = np.stack(imgs, axis=0)
        lr_seq = torch.from_numpy(np.ascontiguousarray(np.transpose(lr_seq, (0, 3, 1, 2)))).float()
        # if self.patch_size != 0:
        #     lr, gt, neighbor, _ = get_patch(ref, gt, lr_seq, self.patch_size,
        #                                     self.upscale_factor, self.nFrames)
        #
        # if self.augmentation:
        #     lr, gt, neighbor, _ = augment(lr, gt, neighbor)
        return lr_seq, gt_path

    def __len__(self):
        return len(self.imgs)

    def __add__(self, other):  # todo
        return data.dataset.ConcatDataset([self, other])

    def generate_names(self, file_name, padding='reflection'):
        """
        padding: replicate | reflection | new_info | circle
        :param crt_i: 当前帧序号
        :param max_n: 视频帧数
        :param N: 序列长度
        :param padding: 补齐模式
        :return: 索引序列 Youku_00000_l_100_00_.npy
        """
        fnl = file_name.split('_')
        max_n, crt_i = fnl[-3:-1]
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
    '''read one image from img_path
    Return img: HWC, BGR, [0,1], numpy
    '''
    img_GT = cv2.imread(img_path)
    img = img_GT.astype(np.float32) / 255.
    return img


def read_seq_imgs(img_seq_path):
    '''read a sequence of images'''
    img_path_l = sorted(glob.glob(img_seq_path + '/*'))
    img_l = [read_image(v) for v in img_path_l]
    # stack to TCHW, RGB, [0,1], torch
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
