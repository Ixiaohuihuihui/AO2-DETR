# -*- coding: utf-8 -*-
# @Time    : 04/01/2022 10:42
# @Author  : Linhui Dai
# @FileName: deformable_detr_refine_r50_16x2_50e_sku.py
# @Software: PyCharm
_base_ = 'deformable_detr_r50_16x2_50e_sku.py'
model = dict(bbox_head=dict(with_box_refine=True))