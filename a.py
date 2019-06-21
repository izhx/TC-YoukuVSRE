import os
import glob
import numpy as np
from data.info_list import CARTOONS

from data.youku import SISRDataset


def main(dir):
    video_paths = glob.glob(f"{dir}/*h_GT")
    avg = list()
    for i, vp in enumerate(video_paths):
        if os.path.basename(vp)[-16:-5] not in CARTOONS:
            continue
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

    yk = SISRDataset(DIR, augment=True, patch_size=128, v_freq=5, preload=True)
    # yk.__getitem__(0)
    # test()
    main(DIR)
    pass
    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'
