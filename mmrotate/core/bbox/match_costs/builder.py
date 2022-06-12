# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.match_costs.builder import MATCH_COST

ROTATED_MATCH_COST = MATCH_COST


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, ROTATED_MATCH_COST, default_args)
