# -*- coding: utf-8 -*-
# @Time    : 16/03/2022 20:39
# @Author  : Linhui Dai
# @FileName: builder.py
# @Software: PyCharm
from mmcv.utils import build_from_cfg
from mmdet.models.utils.builder import TRANSFORMER

ROTATED_TRANSFORMER = TRANSFORMER

def build_tranformer(cfg, default_args=None):
    return build_from_cfg(cfg, ROTATED_TRANSFORMER, default_args)