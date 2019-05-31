import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.EDVR_arch import EDVR
from data.youku import YoukuDataset, read_npy

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
# parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=4e-4, help='Learning Rate. Default=0.0004')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./dataset')
# parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--padding', type=str, default="reflection",
                    help="padding: replicate | reflection | new_info | circle")
parser.add_argument('--model_type', type=str, default='EDVR')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='weights/3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print(opt)


def single_forward(model, imgs_in):
    with torch.no_grad():
        model_output = model(imgs_in)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    return output


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        lr_seq, gt_path = batch[0], batch[1]
        if cuda:
            input = Variable(lr_seq).cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = single_forward(model, lr_seq)
        prediction_f = prediction.data.float().cpu().squeeze(0)
        target = read_npy(gt_path)  # todo 不知道网络出来的啥形式
        loss = criterion(prediction_f, target)
        t1 = time.time()
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader),
                                                                                 loss.data[0], (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def checkpoint(epoch):
    model_out_path = opt.save_folder + str(
        opt.upscale_factor) + 'x_' + opt.model_type + opt.prefix + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


print('===> Loading dataset')
train_set = YoukuDataset(opt.data_dir, opt.upscale_factor, opt.nFrames,
                         opt.data_augmentation, opt.path_size, opt.padding)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'EDVR':
    model = EDVR()
else:
    model = None

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

criterion = nn.L1Loss()  # todo bonnier penalty function

# print('---------- Networks architecture -------------')
# print_network(model)
# print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    # test()  # todo 需加入在验证集检验，满足要求停机

    # todo learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch + 1) % (opt.snapshots) == 0:
        checkpoint(epoch)
