# YoukuSR
“努力，要努力，我要变成万人迷！”

天池2019[阿里巴巴优酷视频增强和超分辨率挑战赛](https://tianchi.aliyun.com/competition/entrance/231711/introduction)自用代码，EDVR、WDSR、ESRGAN三个模型。有参考价值的东西，可能只有`utils/y4m_tools.py`。
## 简介
第一次参加天池，还要期末考试压力好大嘤嘤嘤~

初赛GG了，比赛结束后发现与前排的思路一致，只是没卡没时间哇，2-stage、ensemble、flip都没做，只提交了baseline，不过此次参赛学习到了很多知识。

数据处理是自己手写的，读取y4m文件流，分帧转化为yuv444的；写了几个Dataset，分别给VSR和SISR用，VSR时做了场景分割（[室友](https://github.com/midebuxing)写的机器学习算法，暂时没专门学习场景检测的文献）以及帧序列padding的处理（在[xintao](https://github.com/xinnta)大佬的策略上稍作修改），WDSR修改了一点点代码尝试单通道SR。输出生成y4m和输入差不多。

EDVR、WDSR的model architecture来自[EDVR](https://github.com/xinntao/EDVR)和[wdsr_ntire2018](https://github.com/JiahuiYu/wdsr_ntire2018)，自己写了训练代码。ESRGAN代码全部来自[BasicSR](https://github.com/xinntao/BasicSR)，不得不说大佬所有东西都手写，代码很健壮，造轮子精神可嘉。

自己配置了VMAF并且封装了函数，在`utils/vmaf_tools.py`中。

## 文件结构
```
├── README.md               // readme
├── dataset                 // 数据集
├── data                    // 数据代码
├── utils                   // 通用工具
├── model                   // 模型代码
├── models                  // 模型代码
├── optim                   // 优化器
├── options                 // 配置项
├── vmaf                    // vmaf
├── scripts                 // 
└── ...                     // 
```
## 编码规范
- 尽量符合pycharm的提示，即PEP 8.
- 参考Google的[Python风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/)，或英文原版[Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)。
## 环境依赖等
 `Python 3.6.x` ...
 
 [pytorch使用tensorboard](https://www.endtoend.ai/pytorch-tensorboard).
