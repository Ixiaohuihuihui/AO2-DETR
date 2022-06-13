
## Introduction

### AO2-DETR: Arbitrary-Oriented Object Detection Transformer (https://arxiv.org/abs/2205.12785).

<details open>
<summary><b>Deformable DETR</b></summary>

</details>



## Installation

Please refer to [install.md](docs/en/install.md) for installation guide.

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
There are also tutorials:

* [learn the basics](docs/en/intro.md)
* [learn the config](docs/en/tutorials/customize_config.md)
* [customize dataset](docs/en/tutorials/customize_dataset.md)
* [customize model](docs/en/tutorials/customize_models.md)


## Model Zoo

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [Model Zoo](docs/en/model_zoo.md) page.

<details open>
<summary><b>Supported algorithms:</b></summary>

* [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
* [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
* [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
* [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
* [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
* [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
* [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
* [x] [ReDet](configs/redet/README.md) (CVPR'2021)
* [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
* [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
* [x] [GWD](configs/gwd/README.md) (ICML'2021)
* [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
* [x] [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
* [x] [KFIoU](configs/kfiou/README.md) (arXiv)
* [x] [G-Rep](configs/g_reppoints/README.md) (stay tuned)

</details>

### Model Request

We will keep up with the latest progress of the community, and support more popular algorithms and frameworks. If you have any feature requests, please feel free to leave a comment in [MMRotate Roadmap](https://github.com/open-mmlab/mmrotate/issues/1).

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMRotate. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmrotate2022,
  title={MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author =       {Zhou, Yue and Yang, Xue and Zhang, Gefan and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and Zhang, Wenwei, and Chen, Kai},
  howpublished = {\url{https://github.com/open-mmlab/mmrotate}},
  year =         {2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

* [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
* [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
* [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
* [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab next-generation platform for general 3D object detection.
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
* [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab next-generation action understanding toolbox and benchmark.
* [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
* [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
* [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
* [MMOCR](https://github.com/open-mmlab/mmocr): A comprehensive toolbox for text detection, recognition and understanding.
* [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab next-generation toolbox for generative models.
* [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
* [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
* [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
* [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
* [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
* [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
* [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
