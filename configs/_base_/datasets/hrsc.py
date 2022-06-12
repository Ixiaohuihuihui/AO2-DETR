# -*- coding: utf-8 -*-
# @Time    : 05/04/2022 17:16
# @Author  : Linhui Dai
# @FileName: hrsc.py
# @Software: PyCharm
# dataset settings
dataset_type = 'HRSCDataset'
data_root = '/data2/dailh/HRSC2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'ImageSets/trainval.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        img_prefix=data_root+'FullDataSet/AllImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'ImageSets/trainval.txt',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        img_prefix=data_root+'FullDataSet/AllImages/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classwise=False,
        ann_file=data_root + 'ImageSets/test.txt',
        img_prefix=data_root+'FullDataSet/AllImages/',
        ann_subdir=data_root + 'FullDataSet/Annotations/',
        img_subdir=data_root + 'FullDataSet/AllImages/',
        pipeline=test_pipeline))
