# dataset settings
print("train_chegou")
dataset_type = 'PascalVOCDataset_chegou'
data_root = '../data/chegou/'
#data_root = '../data/pengbu/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 624)
# crop_size = (512, 624)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 624), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(1024, 624), ratio_range=(0.5, 2.0)),
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
        img_scale=(1944,1200),
        # img_scale=(1024, 624),

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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='jpg',
        ann_dir='mask',
        split = 'train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='jpg',
        ann_dir='mask',
        split = 'val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='jpg',
        ann_dir='mask',
        split = 'test.txt',
        pipeline=test_pipeline))