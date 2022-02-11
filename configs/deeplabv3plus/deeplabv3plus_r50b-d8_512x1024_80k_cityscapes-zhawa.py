_base_ = './deeplabv3plus_r50-d8_512x1024_40k_cityscapes-zhawa.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
