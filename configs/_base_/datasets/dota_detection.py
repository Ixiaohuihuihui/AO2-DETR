# -*- coding: utf-8 -*-
# @Time    : 08/10/2021 23:41
# @Author  : Linhui Dai
# @FileName: dota_detection.py
# @Software: PyCharm
# dataset settings
dataset_type = 'DOTA10Dataset'
data_root = '/data2/dailh/dota_coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         with_crowd=False, with_label=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
]
# # test_pipeline, NOTE the Pad's size_divisor is different from the default
# # setting (size_divisor=32). While there is little effect on the performance
# # whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         with_crowd=False, with_label=True),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA_test1024.json',
        img_prefix=data_root + 'test1024/images',
        pipeline=test_pipeline))
