# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .builder import ROTATED_MATCH_COST
from mmcv.ops import box_iou_rotated
# from mmrotate.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma, xy_stddev_pearson_2_xy_sigma
# from mmdet.core.bbox.iou_calculators import gwd_loss
def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss.

    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    """Convert oriented bounding box from the Pearson coordinate system to 2-D
    Gaussian distribution.

    Args:
        xy_stddev_pearson (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.square()
    sigma = torch.stack((var[..., 0], covar, covar, var[..., 1]),
                        dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma

@ROTATED_MATCH_COST.register_module()
class RBBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmrotate.core.bbox.match_costs import RBBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 5)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4, 3], [1, 2, 3, 4, 2]])
         >>> factor = torch.tensor([10, 8, 10, 8, 1])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxya'):
        self.weight = weight
        assert box_format in ['xyxya', 'xywha']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h, a), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_rbboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, w, h, a). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # if self.box_format == 'xywha':
        #     gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        # elif self.box_format == 'xyxya':
        #     bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@ROTATED_MATCH_COST.register_module()
class RotatedIoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = box_iou_rotated(bboxes, gt_bboxes, mode=self.iou_mode, aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight

@ROTATED_MATCH_COST.register_module()
class GaussianIoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """
    BAG_PREP = {
        'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
        'xy_wh_r': xy_wh_r_2_xy_sigma
    }
    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode
        self.preprocess = self.BAG_PREP['xy_stddev_pearson']

    def __call__(self, bboxes, gt_bboxes,  fun='log1p', tau=1.0, alpha=1.0, normalize=True):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        pred = self.preprocess(bboxes)
        target = self.preprocess(gt_bboxes)
        xy_p, Sigma_p = pred
        xy_t, Sigma_t = target

        xy_distance = (xy_p[..., None, :2] - xy_t).square().sum(dim=-1)

        whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance[..., None] + Sigma_t.diagonal(
            dim1=-2, dim2=-1).sum(dim=-1)

        _t = Sigma_p[:, None, ...].matmul(Sigma_t)
        _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _t_det_sqrt = (Sigma_p[:, None, ...].det() * Sigma_t.det()).clamp(0).sqrt()
        whr_distance = whr_distance + (-2) * (
            (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

        distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

        if normalize:
            scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
            distance = distance / scale

        return postprocess(distance, fun=fun, tau=tau) * self.weight
