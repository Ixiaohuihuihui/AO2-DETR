# -*- coding: utf-8 -*-
# @Time    : 15/10/2021 10:19
# @Author  : Linhui Dai
# @FileName: deformable_detr_refine_r50_16x2_50e_dota1_ms.py
# @Software: PyCharm
_base_ = 'deformable_detr_r50_16x2_50e_dota15_ms.py'
model = dict(bbox_head=dict(with_box_refine=True))