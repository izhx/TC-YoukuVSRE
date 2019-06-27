from __future__ import print_function
import os
import time
from datetime import datetime
import math
import argparse
import logging
import pickle
import yaml

import cv2 as cv
import numpy as np
# from skimage.measure.simple_metrics import compare_psnr

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.youku import SISRDataset
from model.WDSR_A import MODEL
from models.modules.RRDBNet_arch import RRDBNet

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

now = datetime.now()
label = f"{opt['model']}-C{opt['channel']}-R{opt[opt['model']]['n_resblocks']}F{opt[opt['model']]['n_feats']}"
tb_dir = f"{opt['log_dir']}/{now.strftime('%m%d-%H%M-')}{label}/"

print('===> Loading dataset')
train_set = SISRDataset(data_dir=opt['data_dir'], augment=opt['augment'],
                        patch_size=opt['patch_size'], v_freq=opt['vFreq'],
                        preload=opt['preload'], norm=False)
data_loader = DataLoader(dataset=train_set, num_workers=opt['hardware']['threads'],
                         batch_size=opt['batch_size'], shuffle=True)
eval_set = SISRDataset(data_dir=opt['eval_dir'], augment=opt['augment'],
                       patch_size=0, v_freq=5)
eval_loader = DataLoader(dataset=eval_set, num_workers=opt['hardware']['threads'],
                         shuffle=True)

print('===> Building model')
if opt['model'] == 'WDSR':
    if opt['channel'] == 3:
        model = MODEL(cuda, n_res=opt['WDSR']['n_resblocks'], n_feats=opt['WDSR']['n_feats'],
                      res_scale=opt['WDSR']['res_scale'], n_colors=3, block_feats=opt['WDSR']['block_feats'],
                      mean=opt['mean']).to(device)
    else:
        model = MODEL(cuda, n_res=opt['WDSR']['n_resblocks'], n_feats=opt['WDSR']['n_feats'],
                      res_scale=opt['WDSR']['res_scale'],
                      n_colors=1, mean=[opt['mean'][opt['channel']]]).to(device)
elif opt['model'] == 'RRDB':
    model = RRDBNet(3, 3, opt['RRDB']['n_feats'], opt['RRDB']['n_resblocks']).to(device)
else:
    model = None

criterion = nn.L1Loss().to(device)

optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
# optimizer = Nadam(model.parameters(), lr=0.00001)
# optimizer = optim.SGD(model.parameters(), lr=opt['lr'], momentum=0.9, weight_decay=1e-4, nesterov=True)

re_avgpool = torch.nn.AvgPool2d((2, 2), stride=(2, 2))

if opt['pre_trained'] and os.path.exists(opt['pre_train_path']):
    model.load_state_dict(torch.load(opt['pre_train_path'], map_location=lambda storage, loc: storage))
    # with open(f"{opt['save_dir']}/optim.pkl", 'rb') as f:
    #     optimizer = pickle.load(f)
    print('Pre-trained SR model is loaded.')


def get_ch(img: torch.Tensor, channel: int):
    if channel == 0:  # Y通道
        return img.index_select(1, torch.LongTensor([channel])).to(device)
    elif 3 > channel > 0:  # U和V
        return re_avgpool(img.index_select(1, torch.LongTensor([channel]))).to(device)
    elif channel == 3:  # 444
        return img.to(device)


def out_rgb(img, path):
    img = img.cpu().squeeze(0).numpy().astype(np.uint8)
    if opt['channel'] < 3:
        img = img[0]
    elif opt['rgb'] == False:
        img = cv.cvtColor(img, cv.COLOR_YUV2RGB)
    cv.imwrite(path, img)
    return


def train(e):
    print(f"===> Epoch {e} Begin: LR: {optimizer.param_groups[0]['lr']}")
    epoch_loss = 0
    model.train()
    for batch_i, batch in enumerate(data_loader):
        t0 = time.time()
        lr, gt = get_ch(batch[0], opt['channel']), get_ch(batch[1], opt['channel'])

        optimizer.zero_grad()
        loss = criterion(model(lr), gt)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        t1 = time.time()

        # 每10个batch画个点用于loss曲线
        if batch_i % 10 == 0:
            print(f"===> Epoch[{e}]({batch_i}/{len(data_loader)}):",
                  f" Loss: {loss.item():.4f} || Timer: {(t1 - t0):.4f} sec.")
            niter = (epoch * len(data_loader) + batch_i) * opt['batch_size']
            with SummaryWriter(log_dir=tb_dir, comment='WDSR')as w:
                w.add_scalar('Train/Loss', loss.item(), niter)

    avg_loss = epoch_loss / len(data_loader)
    with SummaryWriter(log_dir=tb_dir, comment='WDSR')as w:
        w.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], e)
        w.add_scalar('Train/epoch_Loss', avg_loss, e)
    print(f"===> Epoch {e} Complete: Avg. Loss: {avg_loss:.4f}")
    return


def eval_func(e, only=False):
    epoch_loss = 0
    avg_psnr = 0
    if opt['pre_trained'] and only:
        model.load_state_dict(torch.load(opt['pre_train_path']))
    model.eval()
    for batch_i, batch in enumerate(eval_loader):
        t0 = time.time()
        lr, gt = get_ch(batch[0], opt['channel']).to(device), get_ch(batch[1], opt['channel']).to(device)

        with torch.no_grad():
            sr = model(lr)
            _psnr = psnr_tensor(sr, gt)

        loss = criterion(sr, gt)
        t1 = time.time()
        epoch_loss += loss.item()
        avg_psnr += _psnr

        if batch_i % 20 == 0:
            out_rgb(lr, f"./test/{opt['channel']}_{e}_{batch_i}_lr.png")
            out_rgb(sr, f"./test/{opt['channel']}_{e}_{batch_i}_sr.png")
            out_rgb(gt, f"./test/{opt['channel']}_{e}_{batch_i}_gt.png")
            print(f"===> eval({batch_i}/{len(eval_loader)}):  PSNR: {_psnr:.4f}",
                  f" Loss: {loss.item():.4f} || Timer: {(t1 - t0):.4f} sec.")

    avg_psnr /= len(eval_loader)
    avg_loss = epoch_loss / len(eval_loader)
    print(f"===> eval Complete: Avg PSNR: {avg_psnr}, Avg. Loss: {avg_loss:.4f}")
    with SummaryWriter(log_dir=tb_dir, comment='WDSR')as w:
        w.add_scalar('eval/PSNR', avg_psnr, epoch)
        w.add_scalar('eval/LOSS', avg_loss, epoch)
    return avg_psnr


def psnr_tensor(img1: torch.Tensor, img2: torch.Tensor):
    # img1 and img2 have range [0, 255]
    diff = img1 - img2
    mse = torch.mean(diff * diff).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(65025.0 / mse)


def checkpoint(comment=""):
    global opt
    save_path = f"{opt['save_dir']}/{opt['scale']}x_{comment}_{epoch}.pth"
    torch.save(model.state_dict(), save_path)

    with open(args.yaml_path, 'r') as f:
        opt = yaml.load(f)

    opt['pre_train_path'] = save_path
    opt['pre_trained'] = True
    opt['startEpoch'] = epoch + 1
    # with open(f"{opt['save_dir']}/optim.pkl", 'wb') as f:
    #     pickle.dump(optimizer, f)
    with open(args.yaml_path, 'w') as f:
        f.write(yaml.dump(opt))
    print(f"Checkpoint saved to {save_path}")


doEval = opt['only_eval']

if doEval:
    eval_func(-1, opt['only_eval'])
else:
    for epoch in range(opt['startEpoch'], opt['nEpochs'] + 1):
        train(epoch)
        if (epoch + 1) % opt['snapshots'] == 0:
            checkpoint(label)
            eval_func(epoch)
        if (epoch + 1) in opt['lr_step']:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0

# 脚本退出后存储配置
with open(args.yaml_path, 'w') as yf:
    yf.write(yaml.dump(opt))

os.system("bash /root/shutdown.sh")

"""
需要调节的：
- n_resblocks = 16
- n_feats = 64
- 三个通道均值 mean 从数据中来
- lr 的更新
- batch size
- patch size
- v freq  每个视频每epoch抽帧次数
"""
