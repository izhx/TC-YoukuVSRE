from __future__ import print_function
import os
import time
import math
import argparse
import logging
import shutil
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.youku import SISRDataset
from model.WDSR_B import MODEL

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--yaml_path', type=str, default="./settings.yaml", help='配置文件路径')

args = parser.parse_args()
with open(args.yaml_path, 'r') as f:
    opt = yaml.load(f)
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

if not opt['pre_trained']:
    shutil.rmtree(opt['log_dir'])
    os.mkdir(opt['log_dir'])

print('===> Loading dataset')
train_set = SISRDataset(data_dir=opt['data_dir'], augment=opt['augment'],
                        patch_size=opt['patch_size'], v_freq=opt['vFreq'])
data_loader = DataLoader(dataset=train_set, num_workers=opt['hardware']['threads'],
                         batch_size=opt['batch_size'], shuffle=True)
eval_set = SISRDataset(data_dir=opt['eval_dir'], augment=opt['augment'],
                       patch_size=0, v_freq=5)
eval_loader = DataLoader(dataset=eval_set, num_workers=opt['hardware']['threads'],
                         shuffle=True)

print('===> Building model')
model = MODEL(cuda).to(device)
criterion = nn.L1Loss().to(device)

optimizer = optim.Adam(model.parameters(), lr=opt['lr'], betas=(0.9, 0.999), eps=1e-8)

if opt['pre_trained'] and os.path.exists(opt['pre_train_path']):
    model.load_state_dict(torch.load(opt['pre_train_path'], map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')


def train(e):
    epoch_loss = 0
    model.train()
    for batch_i, batch in enumerate(data_loader):
        t0 = time.time()
        lr, gt = batch[0].to(device), batch[1].to(device)

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
            niter = epoch * len(data_loader) + batch_i
            with SummaryWriter(log_dir=opt['log_dir'], comment='WDSR')as w:
                w.add_scalar('Train/Loss', loss.item(), niter)

    print(f"===> Epoch {e} Complete: Avg. Loss: {epoch_loss / len(data_loader):.4f}")
    return


def eval_func():
    epoch_loss = 0
    avg_psnr = 0
    if opt['pre_trained']:
        model.load_state_dict(torch.load(opt['pre_train_path']))
    model.eval()
    for batch_i, batch in enumerate(eval_loader):
        t0 = time.time()
        lr, gt = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            sr = model(lr)
            _psnr = psnr_tensor(sr, gt)

        loss = criterion(sr, gt)
        t1 = time.time()
        epoch_loss += loss.item()
        avg_psnr += _psnr

        print(f"===> eval({batch_i}/{len(data_loader)}):  PSNR: {_psnr:.4f}",
              f" Loss: {loss.item():.4f} || Timer: {(t1 - t0):.4f} sec.")

    avg_psnr /= len(data_loader)
    print(f"===> eval Complete: Avg PSNR: {avg_psnr}",
          f", Avg. Loss: {epoch_loss / len(data_loader):.4f}")
    with SummaryWriter(log_dir=opt['log_dir'], comment='WDSR')as w:
        w.add_scalar('eval/PSNR', avg_psnr, epoch)
    return avg_psnr


def psnr_tensor(img1: torch.Tensor, img2: torch.Tensor):
    # img1 and img2 have range [0, 255]
    diff = img1 - img2
    mse = torch.mean(diff * diff).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def checkpoint(comment=""):
    save_path = f"{opt['save_dir']}{opt['scale']}x_{opt['model']}_{comment}_{epoch}.pth"
    torch.save(model.state_dict(), save_path)
    opt['pre_train_path'] = save_path
    opt['pre_trained'] = True
    with open(args.yaml_path, 'w') as f:
        f.write(yaml.dump(opt))
    print(f"Checkpoint saved to {save_path}")


doEval = opt['only_eval']

if doEval:
    eval_func()
else:
    for epoch in range(opt['startEpoch'], opt['nEpochs'] + 1):
        train(epoch)
        # todo learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (opt['nEpochs'] / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print(f"Learning rate decay: lr={optimizer.param_groups[0]['lr']}")

        if (epoch + 1) % opt['snapshots'] == 0:
            checkpoint(f"R{opt[opt['model']]['n_resblocks']}F{opt[opt['model']]['n_feats']}")
            eval_func()

# 脚本退出后存储配置
with open(args.yaml_path, 'w') as f:
    f.write(yaml.dump(opt))

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
