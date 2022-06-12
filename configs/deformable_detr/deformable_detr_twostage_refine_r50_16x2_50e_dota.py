# -*- coding: utf-8 -*-
# @Time    : 15/10/2021 10:20
# @Author  : Linhui Dai
# @FileName: deformable_detr_twostage_refine_r50_16x2_50e_dota.py
# @Software: PyCharm
_base_ = 'deformable_detr_refine_r50_16x2_50e_dota.py'
model = dict(bbox_head=dict(as_two_stage=True))