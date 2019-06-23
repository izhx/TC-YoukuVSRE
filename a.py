import os
import glob
import numpy as np
from utils.y4m_tools import read_y4m
from data.info_list import CARTOONS

from data.youku import ESRGANDataset


def main(dir):
    video_paths = glob.glob(f"{dir}/*h_GT")
    avg = list()
    for i, vp in enumerate(video_paths):
        # if os.path.basename(vp)[-16:-5] not in CARTOONS:
        #     continue
        paths = glob.glob(f"{vp}/*npy")
        va = np.zeros([len(paths), 3], dtype=np.float64)
        for j, p in enumerate(paths):
            img = np.load(p).astype(np.float64)
            va[j, :] = np.mean(img.mean(axis=0), axis=0)
        avg.append(va.mean(axis=0))
        print(f"{vp} mean: {avg[-1]}")
    avg = np.array(avg)
    np.save('results/car.npy', avg)
    print(avg.mean(axis=0))
    return


def cal_std():
    mean = np.array([99.00332925, 124.7647323, 128.69159715], dtype=np.float64)
    video_paths = glob.glob(f"D:/youku/data/*.y4m")
    all_std = list()
    for vp in video_paths:
        frames, _ = read_y4m(vp)
        std_l = np.zeros([len(frames), 3], dtype=np.float64)
        for j in range(len(frames)):
            diff = frames[j].astype(np.float64) - mean
            std = np.sqrt(diff * diff)
            std_l[j, :] = np.mean(std.mean(axis=0), axis=0)
        all_std.append(std_l.mean(axis=0))
        print(f"{vp} mean: {all_std[-1]}")
    all_std = np.array(all_std)
    np.save('std.npy', all_std)
    print(all_std.mean(axis=0))
    print((all_std / 255).mean(axis=0))
    return


def stat_info():
    video_paths = glob.glob(f"D:/youku/data/*.y4m")
    all_mm = list()
    for vp in video_paths:
        frames, _ = read_y4m(vp)
        std_max = np.zeros([len(frames), 3], dtype=np.float64)
        std_min = np.zeros([len(frames), 3], dtype=np.float64)
        for j in range(len(frames)):
            std_max[j, :] = np.max(frames[j].max(axis=0), axis=0)
            std_min[j, :] = np.min(frames[j].min(axis=0), axis=0)
        all_mm.append(std_max.max(axis=0))
        all_mm.append(std_min.min(axis=0))
        print(f"{vp} min: {all_mm[-1]}, max: {all_mm[-2]}")
    all_mm = np.array(all_mm)
    np.save('stat.npy', all_mm)
    print(all_mm.min(axis=0))
    print(all_mm.max(axis=0))
    print((all_mm / 255).min(axis=0))
    print((all_mm / 255).max(axis=0))
    return


if __name__ == '__main__':
    DIR = r"D:/DATA/train"

    # ds = glob.glob(f"{DIR}/*_l")
    # for dp in ds:
    #     fpl = glob.glob(f'{dp}/*.npy')
    #     for i, fp in enumerate(fpl):
    #         if i % 10 != 0:
    #             os.remove(fp)
    #             os.remove(fp.replace('_l', '_h_GT'))
    #         else:
    #             print(fp)
    # stat_info()
    cfg = {'data_dir': DIR,
           'scale': 4,
           'v_freq': 5,
           'n_workers': 6,  # per GPU
           'batch_size': 16,
           'patch_size': 256,
           'augmentation': True,
           'color': 'RGB'}
    yk = ESRGANDataset(cfg)
    yk.__getitem__(0)
    # test()
    main(DIR)
    pass
    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'

"""
均值
99.00332925, 124.7647323 , 128.69159715
0.38824835, 0.48927346, 0.50467293

标准差：
[51.16912088  9.29543705  9.23474285]
[0.20066322 0.03645269 0.03621468]

最大最小值
[0. 0. 8.]
[255. 235. 255.]
[0.         0.         0.03137255]
[1.         0.92156863 1.        ]

"""
