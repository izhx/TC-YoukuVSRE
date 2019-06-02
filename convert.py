import argparse

from utils import y4m_tools

DIR = "./dataset/train"
parser = argparse.ArgumentParser(description='Youku data convert tool')
parser.add_argument('--data_dir', type=str, default=DIR, help="数据文件夹，相对路径")
parser.add_argument('--cut', type=bool, default=False, help="转场切割，没搞完，不能用。")

if __name__ == '__main__':
    opt = parser.parse_args()
    y4m_tools.convert(opt.data_dir, opt.cut)
    pass
    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'
