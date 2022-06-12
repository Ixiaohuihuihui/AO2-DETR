# -*- coding: utf-8 -*-
# @Time    : 15/10/2021 10:20
# @Author  : Linhui Dai
# @FileName: deformable_detr_twostage_refine_r50_16x2_50e_dota1_ms.py
# @Software: PyCharm
_base_ = 'deformable_detr_refine_r50_16x2_50e_dota15_ms.py'
model = dict(backbone=dict(
    depth=101,
    init_cfg=dict(type='Pretrained',
                  checkpoint='torchvision://resnet101')),
    bbox_head=dict(as_two_stage=True))
data = dict(
    samples_per_gpu=8, workers_per_gpu=4)
