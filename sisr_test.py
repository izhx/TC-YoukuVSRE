from __future__ import print_function
import os
import time
import argparse
import logging
import yaml
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.y4m_tools import read_y4m, save_y4m
from model.WDSR_B import MODEL

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--yaml_path', type=str, default="./settings.yaml", help='配置文件路径')

args = parser.parse_args()
with open(args.yaml_path, 'r') as yf:
    opt = yaml.load(yf)
cudnn.benchmark = True
cuda = opt['hardware']['cuda']
logger = logging.getLogger('base')

print(opt)

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt['hardware']['seed'])
if cuda:
    torch.cuda.manual_seed(opt['hardware']['seed'])
device = torch.device("cuda" if cuda else "cpu")

print('===> Building model')
model = MODEL(cuda, n_res=opt['WDSR']['n_resblocks'], n_feats=opt['WDSR']['n_feats'],
              res_scale=opt['WDSR']['res_scale']).to(device)

if opt['pre_trained'] and os.path.exists(opt['pre_train_path']):
    model.load_state_dict(torch.load(opt['pre_train_path'], map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')

avgpool = torch.nn.AvgPool2d((2, 2), stride=(2, 2))


def single_test(video_path):
    fac = opt['scale']
    print(f'Processing: {video_path}')
    t0 = time.time()
    frames, header = read_y4m(video_path)
    header = header.split()
    vid = os.path.basename(video_path)[:-6]

    size = np.array(frames[0].shape[:2])
    pad_size = (np.ceil(size / 4) * 4 - size).astype(np.int)
    hr_size, hr_pad = size * fac, pad_size * fac

    # 预处理
    frames = np.stack(frames, axis=0)
    frames = np.pad(frames, ((0, 0), (pad_size[0], pad_size[1]), (0, 0), (0, 0)), 'constant',
                    constant_values=(0, 0))

    # 后9/10抽帧
    if int(vid[6:]) > 204:
        thin_frames = list()
        for i, f in enumerate(frames):
            if i % 25 == 0:
                thin_frames.append(f)
        frames = thin_frames

    def convert_channel(ch: torch.tensor):
        ch = ch.numpy().flatten()
        ch = (ch * 255).round().astype(np.uint8)
        # Important. Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        return ch

    hr_frames = list()
    for lr in frames:
        lr_in = torch.from_numpy(np.ascontiguousarray(lr.transpose((2, 0, 1)))).float().to(device)
        # 单帧超分
        with torch.no_grad():
            output = model(lr_in)[0]
        output_f = output.data.float().cpu()
        output_f = output_f[:, hr_pad[0]:, hr_pad[1]:]
        prediction_pool = avgpool(output_f)
        # 给出像素
        y = convert_channel(output_f[0, :, :])
        u = convert_channel(prediction_pool[1, :, :])
        v = convert_channel(prediction_pool[2, :, :])
        hr_frames.append(np.concatenate((y, u, v)))

    header[1] = b'W' + str(hr_size[1]).encode()
    header[2] = b'H' + str(hr_size[0]).encode()
    save_path = f'{opt["result_dir"]}/{os.path.basename(video_path).replace("_l", "_h_Res")}'
    header = b' '.join(header) + b'\n'

    # 后9/10抽帧存储
    if int(vid[6:]) > 204:
        save_y4m(hr_frames, header, save_path.replace('_h', '_h_Sub25'))
    else:  # 存完整的
        save_y4m(hr_frames, header, save_path)
    t1 = time.time()
    print(f'One video saved: {save_path}, timer: {(t1 - t0):.4f} sec.')
    return


def test():
    test_paths = glob.glob(f"{opt['test_dir']}/*_l.y4m")
    for vp in test_paths:
        single_test(vp)
    return


if __name__ == '__main__':
    test()
