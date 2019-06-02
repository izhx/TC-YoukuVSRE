import torch
from torch.utils.data import DataLoader
from data.youku import YoukuDataset

if __name__ == '__main__':
    # yk = YoukuDataset("../dataset/train", 4, 5, True, 31, "new_info")
    DIR = "../dataset/train"
    train_set = YoukuDataset("./dataset/train", 4, 7, True, 0, 'replicate', v_freq=5)
    data_loader = DataLoader(dataset=train_set, batch_size=2,
                             shuffle=True, num_workers=0,
                             collate_fn=train_set.collate_fn)
    for batch_i, (lr_seq, gt) in enumerate(data_loader):
        batches_done = len(data_loader) * 1 + batch_i
        print(batches_done)
    pass
    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'
