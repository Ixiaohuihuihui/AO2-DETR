# -*- coding: utf-8 -*-
# @Time    : 08/01/2022 17:26
# @Author  : Linhui Dai
# @FileName: deformable_detr_twostage_refine_r152_16x2_50e_dota15_ms.py
# @Software: PyCharm
_base_ = 'deformable_detr_refine_r50_16x2_50e_dota15_ms.py'
model = dict(backbone=dict(
    depth=152,
    init_cfg=dict(type='Pretrained',
                  checkpoint='torchvision://resnet152')),
    bbox_head=dict(as_two_stage=True))
data = dict(
    samples_per_gpu=8, workers_per_gpu=4)