print("chegou")
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8_chegou.py',
    '../_base_/datasets/train_chegou.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
