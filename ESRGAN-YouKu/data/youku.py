import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data


class YoukuDataset(data.Dataset):
    def __init__(self, opt):
        #data_dir, upscale_factor, augmentation, patch_size, padding, v_freq=1, cut=True
        super(YoukuDataset, self).__init__()
        data_dir=opt['data_dir']
        self.upscale_factor = opt['upscale_factor']
        self.augmentation = opt['augmentation']
        self.patch_size = opt['patch_size']
        self.data_dir = opt['data_dir']

        v_freq=opt['v_freq'] if opt['v_freq'] else 1
        cut = opt['cut'] if opt['cut'] else True

        if cut:
            #筛选出动画
            self.paths = [os.path.normpath(v) for v in glob.glob(f"{data_dir}/*_l*") if os.path.isdir(v)] * v_freq
        else:
            self.paths = [os.path.normpath(v) for v in glob.glob(f"{data_dir}/*_l")] * v_freq
        print(self.paths)
        return

    def __getitem__(self, index):
        frame_paths = glob.glob(f"{self.paths[index]}\\*.npy")
        # 随机抽帧
        ref_id = random.randint(0, len(frame_paths)-1)
        # 取数据
        #print(f'ref_id:{ref_id}, len:{len(frame_paths)}')
        ref_path = frame_paths[ref_id]
        gt_path = f"{ref_path}".replace('_l', '_h_GT')  # 取GT
        hr = read_npy(gt_path)
        imgs = read_npy(ref_path)  # lr 序列

        if self.augmentation:
            imgs, hr, _ = augment(imgs, hr)

        if self.patch_size == 0:  # 不patch，要把图像补齐到4的倍数
            pad_size = (np.ceil(np.array(imgs.shape)[0:2] / 4) * 4 - np.array(imgs.shape)[0:2]).astype(np.int)
            imgs = np.pad(imgs, ((0, 0), (pad_size[0], pad_size[1]), (0, 0), (0, 0)), 'constant',
                          constant_values=(0, 0))
            hr = np.pad(hr, ((0, 0), (pad_size[0] * 4, pad_size[1] * 4), (0, 0)), 'constant',
                        constant_values=(0, 0))
        else:  # patch
            imgs, hr = get_patch(imgs, hr, self.patch_size)
        # to tensor
        lr_seq = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (2, 0, 1)))).float()
        gt = torch.from_numpy(np.ascontiguousarray(np.transpose(hr, (2, 0, 1)))).float()
        return {'LR': lr_seq, 'HR': gt, 'LR_path': ref_path, 'HR_path': gt_path}


    def collate_fn(self, batch):
        lr_seq, gt = list(zip(*batch))
        return torch.stack(lr_seq), torch.stack(gt)

    def __len__(self):
        return len(self.paths)

    def __add__(self, other):
        return data.dataset.ConcatDataset([self, other])



class Error(Exception):
    pass


def read_npy(path):
    x=np.load(path).astype(np.float32)
    return  (x-np.min(x))/(np.max(x)-np.min(x))


def augment(lr_seq, hr, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        hr = cv2.flip(hr, 1)
        lr_seq = cv2.flip(lr_seq, 1)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            hr = cv2.flip(hr, 0)
            lr_seq = cv2.flip(lr_seq, 0)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            hr = rotate(hr, 180)
            lr_seq = rotate(lr_seq, 180)
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
    (h, w, _) = lr_seq.shape
    if patch_size > w or patch_size > h:
        raise ValueError('图像比patch小')
    elif patch_size % 4 != 0:
        raise ValueError('patch size 不是4的倍数')

    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    lr_seq = lr_seq[y:y + patch_size, x:x + patch_size, :]
    hr = hr[y << 2:(y + patch_size) << 2, x << 2:(x + patch_size) << 2, :]
    return lr_seq, hr
