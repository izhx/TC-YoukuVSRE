from __future__ import print_function
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data.youku import SISRDataset
from model.WDSR_B import MODEL
from utils.util import calculate_psnr

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--scale', type=int, default=4, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', type=bool, default=False, action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--pretrained_sr', default='weights/3x_edvr_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

# Hardware specifications
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
# Data specifications
parser.add_argument('--data_dir', type=str, default='./dataset', help='dataset directory')
parser.add_argument('--patch_size', type=int, default=192, help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--chop', type=bool, default=False, action='store_true', help='enable memory-efficient forward')
parser.add_argument('--augmentation', type=bool, default=False)
parser.add_argument('--v_freq', type=int, default=15, help='每个视频每代出现次数')

# Model specifications
parser.add_argument('--model', default='WDSR', help='model name')
parser.add_argument('--pre_trained', type=bool, default=False, help='pre-trained model directory')
parser.add_argument('--pre_train_path', type=str, default='', help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=128, help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')
# parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
# parser.add_argument('--dilation', type=bool, default=False, action='store_true', help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--r_mean', type=float, default=0.4488, help='Mean of R Channel')
parser.add_argument('--g_mean', type=float, default=0.4371, help='Mean of G channel')
parser.add_argument('--b_mean', type=float, default=0.4040, help='Mean of B channel')
parser.add_argument('--block_feats', type=int, default=512, help='Mean of B channel')

opt = parser.parse_args()
cudnn.benchmark = True
cuda = opt.cuda

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = SISRDataset(opt.data_dir, opt.scale, opt.augmentation, opt.patch_size, opt.v_freq)
data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                         shuffle=True)

print('===> Building model')
model = MODEL(opt).to(device)
criterion = nn.L1Loss().to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')


def train(e):
    epoch_loss = 0
    model.train()
    for batch_i, batch in enumerate(data_loader, 1):
        t0 = time.time()
        lr, gt = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(lr), gt)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        t1 = time.time()

        print(f"===> Epoch[{e}]({batch_i}/{len(data_loader)}):",
              f" Loss: {loss.item():.4f} || Timer: {(t1 - t0):.4f} sec.")

    print(f"===> Epoch {e} Complete: Avg. Loss: {epoch_loss / len(data_loader):.4f}")
    return


def eval_func():
    epoch_loss = 0
    t_psnr = 0
    model.load_state_dict(torch.load(opt.save_folder + '4x_EDVRyk_epoch_54.pth'))
    model.eval()
    for batch_i, batch in enumerate(data_loader, 1):
        t0 = time.time()
        lr, gt = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            sr = model(lr)

        loss = criterion(sr, gt)
        t1 = time.time()
        epoch_loss += loss.item()

        y_lr, y_gt = sr[:, 0, :, :], gt[:, 0, :, :]
        y_lr, y_gt = y_lr.cpu().numpy() * 255, y_gt.cpu().numpy() * 255
        # 只计算Y通道PSNR
        avg_psnr = calculate_psnr(y_lr, y_gt)
        t_psnr += avg_psnr

        print(f"===> eval({batch_i}/{len(data_loader)}):  PSNR: {avg_psnr:.4f}",
              f" Loss: {loss.item():.4f} || Timer: {(t1 - t0):.4f} sec.")

    t_psnr /= len(data_loader)
    print(f"===> eval Complete: Avg PSNR: {t_psnr}",
          f", Avg. Loss: {epoch_loss / len(data_loader):.4f}")
    return t_psnr


def checkpoint(epoch_now):
    model_out_path = opt.save_folder + str(opt.scale) + \
                     "x_WDSRyk_epoch_{}.pth".format(epoch_now)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


doEval = False

if doEval:
    eval_func()
else:
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(epoch)
        # todo learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (opt.nEpochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print(f"Learning rate decay: lr={optimizer.param_groups[0]['lr']}")

        if (epoch + 1) % opt.snapshots == 0:
            checkpoint(epoch)

"""
需要调节的：
- n_resblocks = 16
- n_feats = 128
- block_feats = 512
- 三个通道均值 mean 从数据中来
- lr 的更新
- batch size
- patch size
- v freq  每个视频每epoch抽帧次数
"""
