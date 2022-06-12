# -*- coding: utf-8 -*-
# @Time    : 22/04/2022 09:43
# @Author  : Linhui Dai
# @FileName: sku.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time    : 22/04/2022 09:33
# @Author  : Linhui Dai
# @FileName: sku.py
# @Software: PyCharm
import itertools
import logging
import os.path as osp
import os
from mmcv.parallel import DataContainer as DC
import tempfile
from collections import OrderedDict
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS
from mmdet.datasets import CustomDataset
# from mmdet.datasets.utils import to_tensor, random_scale


@ROTATED_DATASETS.register_module()
class SKUDataset(CustomDataset):
    CLASSES = ('0', )
    PALETTE = [(255, 255, 0)]
    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(SKUDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        # data_infos = []
        # data_info = {}
        # data_info['filename'] = img_info
        # data_info['ann'] = {}
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # x1, y1, w, h, theta = ann['rbbox']
            poly = ann['segmentation']
            try:
                x, y, w, h, a = poly2obb_np(poly, self.version)
            except:  # noqa: E722
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            # [x, y, x, y]
            bbox = [x, y, w, h, a]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_polygons.append(poly)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_polygons = np.array(gt_polygons, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_polygons = np.zeros((0, 8), dtype=np.float32)

        if gt_polygons_ignore:
            gt_bboxes_ignore = np.array(
                gt_bboxes_ignore, dtype=np.float32)
            gt_labels_ignore = np.array(
                gt_labels_ignore, dtype=np.int64)
            gt_polygons_ignore = np.array(
                gt_polygons_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros(
                (0, 5), dtype=np.float32)
            gt_labels_ignore = np.array(
                [], dtype=np.int64)
            gt_polygons_ignore= np.zeros(
                (0, 8), dtype=np.float32)
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            polygons=gt_polygons,
            bboxes_ignore=gt_bboxes_ignore,
            labels_ignore=gt_labels_ignore,
            polygons_ignore=gt_polygons_ignore)
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                 scale_ranges=None,
                 use_07_metric=True,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results

