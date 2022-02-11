## 使用 [MMsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.18.0) 的2D语义分割网络测量物体的尺寸

### 搭建环境：
系统：Windows10

Mmcv-full版本为1.3.15

Torch1.8.0

Torchvision0.9

mmsegmentation 0.18.0

#### 步骤1：
在anaconda中创建虚拟环境，python版本为3.6

#### 步骤2：
安装CUDA、Pytorch

#### 步骤3：
安装mmcv-full

`pip install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html`

#### 测试：
在`mmsegmentation`目录下新建文件夹`checkpoints`

`mkdir checkpoints`

`cd checkpoints`

`下载pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth`

地址：
https://github.com/open-mmlab/mmsegmentation/tree/master/configs/pspnet

cd ..\

cd demo

运行
`python image_demo.py demo.png ..\configs\pspnet\pspnet_r50-d8_512x1024_40k_cityscapes.py 
  ..\checkpoints\pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth`

## 使用自己的数据集进行训练：

### 数据集准备：
使用labelme标注数据，然后将标注的json文件转换成png,生成两个文件夹:原图文件夹jpg和掩膜文件夹png
(图像位深度为8，需要检查掩膜文件夹中的png文件是否图像位深为8，若有不为8的数据，需要进行转换,
转换后注意看标签的像素值是否发生变化）

### 配置相关参数：

  相关的目录树
```
  ├─checkpoints(下载预训练权重,地址和测试时候的一样)
  ├─configs
  │  ├─deeplabv3plus
  │                 └─deeplabv3plus_r50-d8_512x1024_40k_cityscapes_zhawa.py(复制deeplabv3plus_r50-d8_512x1024_40k_cityscapes改名)
  │  └─_base_
  │      ├─datasets
  │      │         └─train_pratice.py(复制pascal_voc12.py改名)
  │      ├─models           
  │      └─schedules
  │                └─schedule_20k.py(修改)
  ├─data
  │  ├─zhawa(新建)
  │     ├─zhawa_result(网络训练好的权重保存地址)
  │     ├─zhawa_test(测试图片分割预测结果保存位置，及距离测量结果保存)
  │     ├─jpj(原图)
  │     ├─png(mask)
  │     ├─zdf
  │             └─HD_zhawa(存放原始的zdf文件，其中的zdf文件名要与test.txt中的文件名一一对应)
  │     └─splits(分离训练集、验证集和测试集)
  │             ├─train.txt
  │             ├─val.txt
  │             └─test.txt
  ├─mmseg(修改以后将文件夹复制到 .../envs\mmdetection\Lib\site-packages，及环境中)
  │  ├─datasets
  │     ├─zhawa_voc.py(复制voc.py改名)
  │     └─__init__.py(修改)
```
### 修改过的地方
`deeplabv3plus_r50-d8_512x1024_40k_cityscapes_zhawa.py`内容：
```
_base_ = [
      '../_base_/models/deeplabv3plus_r50-d8.py',
      '../_base_/datasets/train_pratice.py', '../_base_/default_runtime.py', //修改train_pratice.py
      '../_base_/schedules/schedule_40k.py'
  ]
```
`train_pratice.py`的内容
```
# dataset settings
dataset_type = 'PascalVOCDataset_zhawa' //制作数据集时的类名字
data_root = '../data/zhawa/'  //数据集的路径
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 768), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 768),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, //通过这个参数设置batch size
    workers_per_gpu=2, //
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='jpj',
        ann_dir='png',
        split = 'splits/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='jpj',
        ann_dir='png',
        split = 'splits/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='jpj',
        ann_dir='png',
        split = 'splits/test.txt',
        pipeline=test_pipeline))


```
`schedule_40k.py`的内容
```
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000) //设置训练时网络迭代的epoch
checkpoint_config = dict(by_epoch=False, interval=200) //训练时每隔200epoch，网络保存一次权重
evaluation = dict(interval=200, metric='mIoU') //训练时每隔200epoch，网络使用验证集进行验证一次

```
`train_pratice.py`的内容
```
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset_zhawa(CustomDataset):  //将PascalVOCDataset 改为PascalVOCDataset_zhawa
    """train_db dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    CLASSES = ('background', 'tielian', 'ruanguan')  //设置为自己数据集的标签名字，注意要加上background

    PALETTE = [[128, 128, 128], [128, 0, 0], [0, 128, 0]]  //给每一个标签设置一个颜色

    def __init__(self, **kwargs):
        super(PascalVOCDataset_zhawa, self).__init__(  //将class的名称换为PascalVOCDataset_zhawa
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        assert osp.exists(self.img_dir)
```
`__init__.py`的内容
```
from .ade import ADE20KDataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .chase_db1 import ChaseDB1Dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .drive import DRIVEDataset
from .hrf import HRFDataset
from .pascal_context import PascalContextDataset
from .stare import STAREDataset
from .voc import PascalVOCDataset
from .train_db import TrainDBDataset
from .train_pratice import TrainPDataset
from .zhawa_voc import PascalVOCDataset_zhawa

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset',
    'PascalVOCDataset', 'ADE20KDataset', 'PascalContextDataset',
    'ChaseDB1Dataset', 'DRIVEDataset', 'HRFDataset', 'STAREDataset',
    'TrainDBDataset','TrainPDataset', 'PascalVOCDataset_zhawa'
]


生成split的代码
import numpy as np
filename_val='val.txt'
filename_train='train.txt'
train_list=[]
for i in range(1,1203):
    train_list.append(i)
    np.random.shuffle(train_list)
print(len(train_list))
with open(filename_train,'w') as file_object_train:
    for n in range(0,1051):
        numper_train=train_list[n]
        s_train=str(numper_train)
        file_object_train.write(s_train.zfill(4))
        file_object_train.write("\n")
with open(filename_val,'w') as file_object_val:
    for n in range(1051,1203):
        numper_val=train_list[n]
        s_val=str(numper_val)
        file_object_val.write(s_val.zfill(4))
        file_object_val.write("\n")

```
### 运行`tools\train.py`文件等待训练。。。。。。。。。。。。。。。


### 修改路径，运行`tools\xxx_test`文件，进行物体尺寸测量。