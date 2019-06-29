import os
import sys

from collections import OrderedDict

import numpy as np

import vmaf
from vmaf.config import VmafConfig
from vmaf.core.asset import Asset
from vmaf.core.quality_runner import VmafQualityRunner, PsnrQualityRunner
from vmaf.tools.misc import get_file_name_without_extension
from vmaf.tools.stats import ListStats

from utils.y4m_tools import save_yuv, read_y4m

import matplotlib

matplotlib.use('Agg')

__copyright__ = "Copyright 2016-2019, Netflix, Inc."
__license__ = "Apache, Version 2.0"

FMTS = ['yuv420p', 'yuv422p', 'yuv444p', 'yuv420p10le', 'yuv422p10le', 'yuv444p10le']
POOL_METHODS = ['mean', 'harmonic_mean', 'min', 'median', 'perc5', 'perc10', 'perc20']

MODEL_PATH = 'C:/Workspace/vmaf-master/model/vmaf_v0.6.1.pkl'  # vmaf官方模型

VMAF_PROGRAM_PATH = 'C:/Workspace/vmaf-master/x64/Release'  # 编译好的程序文件放在一起

vmaf.ExternalProgram.vmaf = f"{VMAF_PROGRAM_PATH}/vmaf.exe"  # linux去掉.exe
vmaf.ExternalProgram.psnr = f"{VMAF_PROGRAM_PATH}/psnr.exe"
vmaf.ExternalProgram.ssim = f"{VMAF_PROGRAM_PATH}/ssim.exe"
vmaf.ExternalProgram.moment = f"{VMAF_PROGRAM_PATH}/moment.exe"
vmaf.ExternalProgram.ms_ssim = f"{VMAF_PROGRAM_PATH}/ms_ssim.exe"
vmaf.ExternalProgram.vmafossexec = f"{VMAF_PROGRAM_PATH}/vmafossexec.exe"

# 临时文件存储
if not os.path.exists('./temp'):
    os.makedirs('./temp')


def vmaf_score(ref_path, dis_path, width=1920, height=1080, fmt='yuv420p', pool_method='mean'):
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative, but are {w} and {h}".format(w=width, h=height))

    if fmt not in FMTS:
        raise ValueError("不支持的类型！")

    if not (pool_method is None
            or pool_method in POOL_METHODS):
        raise ValueError('--pool can only have option among {}'.format(', '.join(POOL_METHODS)))

    show_local_explanation = False

    enable_conf_interval = False

    if show_local_explanation and enable_conf_interval:
        print('cannot set both --local-explain and --ci flags')
        return 2

    asset = Asset(dataset="cmd",
                  content_id=abs(hash(get_file_name_without_extension(ref_path))) % (10 ** 16),
                  asset_id=abs(hash(get_file_name_without_extension(ref_path))) % (10 ** 16),
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': width, 'height': height, 'yuv_type': fmt}
                  )
    assets = [asset]

    if show_local_explanation:
        from vmaf.core.quality_runner_extra import VmafQualityRunnerWithLocalExplainer
        runner_class = VmafQualityRunnerWithLocalExplainer
    elif enable_conf_interval:
        from vmaf.core.quality_runner import BootstrapVmafQualityRunner
        runner_class = BootstrapVmafQualityRunner
    else:
        runner_class = VmafQualityRunner

    optional_dict = {'model_filepath': MODEL_PATH}

    runner = runner_class(
        assets, None, fifo_mode=True,
        delete_workdir=True,
        result_store=None,
        optional_dict=optional_dict,
        optional_dict2=None,
    )

    # run
    runner.run()
    result = runner.results[0]

    # pooling
    if pool_method == 'harmonic_mean':
        result.set_score_aggregate_method(ListStats.harmonic_mean)
    elif pool_method == 'min':
        result.set_score_aggregate_method(np.min)
    elif pool_method == 'median':
        result.set_score_aggregate_method(np.median)
    elif pool_method == 'perc5':
        result.set_score_aggregate_method(ListStats.perc5)
    elif pool_method == 'perc10':
        result.set_score_aggregate_method(ListStats.perc10)
    elif pool_method == 'perc20':
        result.set_score_aggregate_method(ListStats.perc20)
    else:  # None or 'mean'
        pass

    # local explanation
    if show_local_explanation:
        runner.show_local_explanations([result])

        # if save_plot_dir is None:
        #     DisplayConfig.show()
        # else:
        #     DisplayConfig.show(write_to_dir=save_plot_dir)

    return result.to_dict()


def psnr_score(ref_path, dis_path, width=1920, height=1080, fmt='yuv420p', pool_method='mean'):
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative, but are {w} and {h}".format(w=width, h=height))

    if fmt not in FMTS:
        raise ValueError("不支持的类型！")

    if not (pool_method is None
            or pool_method in POOL_METHODS):
        raise ValueError('--pool can only have option among {}'.format(', '.join(POOL_METHODS)))

    show_local_explanation = False

    enable_conf_interval = False

    if show_local_explanation and enable_conf_interval:
        print('cannot set both --local-explain and --ci flags')
        return 2

    asset = Asset(dataset="cmd", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=ref_path,
                  dis_path=dis_path,
                  asset_dict={'width': width, 'height': height, 'yuv_type': fmt}
                  )
    assets = [asset]

    runner_class = PsnrQualityRunner

    runner = runner_class(
        assets, None, fifo_mode=True,
        delete_workdir=True,
        result_store=None,
        optional_dict=None,
        optional_dict2=None,
    )

    # run
    runner.run()
    result = runner.results[0]

    # pooling
    if pool_method == 'harmonic_mean':
        result.set_score_aggregate_method(ListStats.harmonic_mean)
    elif pool_method == 'min':
        result.set_score_aggregate_method(np.min)
    elif pool_method == 'median':
        result.set_score_aggregate_method(np.median)
    elif pool_method == 'perc5':
        result.set_score_aggregate_method(ListStats.perc5)
    elif pool_method == 'perc10':
        result.set_score_aggregate_method(ListStats.perc10)
    elif pool_method == 'perc20':
        result.set_score_aggregate_method(ListStats.perc20)
    else:  # None or 'mean'
        pass

    return result.to_dict()


def compute_score(srs: list, gts: list, height=1920, width=1080, fmt='yuv420p', pool_method='mean'):
    ref, dis = './temp/ref.yuv', './temp/dif.yuv'
    try:
        save_yuv(srs, dis), save_yuv(gts, ref)
    except:
        raise SystemError("文件写入失败！")
    psnr = psnr_score(ref, dis, width, height, fmt, pool_method)
    vmfs = vmaf_score(ref, dis, width, height, fmt, pool_method)

    result = OrderedDict()
    result['aggregate'] = OrderedDict({'PSNR_score': psnr['aggregate']['PSNR_score'],
                                       'VMAF_score': vmfs['aggregate']['VMAF_score'],
                                       'method': pool_method})
    frames = list()
    for i in range(len(srs)):
        p = psnr['frames'][i]['PSNR_score']
        v = vmfs['frames'][i]['VMAF_score']
        frames.append(OrderedDict({'frameNum': i, 'PSNR_score': p, 'VMAF_score': v}))
    result['frames'] = frames
    return result


def PSNR_VMAF(srs: list, gts: list, height=1920, width=1080, fmt='yuv420p', pool_method='mean'):
    result = compute_score(srs, gts, height, width, fmt, pool_method)
    return result['aggregate']['PSNR_score'], result['aggregate']['VMAF_score']


if __name__ == '__main__':
    ref_p = 'C:/Users/z/Desktop/contrast/Youku_00182_h_GT.y4m'
    dis_p = 'C:/Users/z/Desktop/contrast/Youku_00182_h_Res.y4m'
    ref, _ = read_y4m(ref_p, mode='420p')
    dis, _ = read_y4m(dis_p, mode='420p')
    res = compute_score(dis, ref)
    print(str(res))
    print(res['aggregate']['PSNR_score'] * 0.8 + res['aggregate']['VMAF_score'] * 0.2)
    print(0)

'''
依赖库：

libsvm-3.23-cp36-cp36m-win_amd64.whl  # 没找到linux的

numpy (>=1.12.0)
scipy (>=0.17.1)
matplotlib (>=2.0.0)
pandas (>=0.19.2)
scikit-learn (>=0.18.1)
scikit-image (>=0.13.1)
h5py (>=2.6.0)
sureal (>=0.1.1)

'''
