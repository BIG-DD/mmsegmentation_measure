_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_sashaguan.py',
    '../_base_/datasets/train_shaguan.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
