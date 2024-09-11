Hi, 感谢大家对我工作的兴趣。这是我21年做的工作，当时没有保持开源和保存模型的好习惯，所以后面做实验把前面的代码给覆盖了。另外，我们实验室的服务器出了些意外，不小心被消防管道的水淹了。所以这份论文的代码没有保存下来，这是我的失误，没有保存模型和代码到本地。希望大家谅解。不过，我的这篇论文是第一篇开源的用DETR框架来做旋转目标检测的，大家可以用这份代码来作为Baseline，在上面修改自己的DETR系列的代码。

我后续也养成了保存模型的习惯，后面发表的论文，都已经把代码和模型上传了。也希望大家也能保持及时保存代码和模型的习惯。

祝各位科研顺利！

## Introduction

### AO2-DETR: Arbitrary-Oriented Object Detection Transformer (https://arxiv.org/abs/2205.12785).
In this paper, we propose an end-to-end transformer-based detector AO2-DETR for arbitrary-oriented object detection.
The proposed AO2-DETR comprises dedicated components to address AOOD challenges, including an oriented proposal generation mechanism, an adaptive oriented proposal refinement module, and a rotation aware set matching loss in order to accurately detect oriented objects in images. The encoderdecoder architecture transforms the oriented proposals (served as object queries) into each corresponding object, which eliminates the need for hand-designed components and complex pre/post-processing. Our approach achieves state-of-theart performance compared to recently anchor-free and singlestage methods on the oriented datasets (DOTA, SKU110K-R and HRSC2016 datasets). We validate that the transformer can enable adaptive receptive fields for oriented objects, thus it can deal with oriented and irregular placed objects naturally.
Furthermore, we hope that this encoder-decoder paradigm will promote future works in oriented object detection.
![Snipaste_2022-06-17_11-58-45](https://user-images.githubusercontent.com/26215859/174222183-2de9fe00-8dd2-4535-8427-d9c385f145f8.png)
<img width="474" alt="image" src="https://user-images.githubusercontent.com/26215859/192183273-e86ee8f0-e96e-4251-a4c3-20885cb497f9.png">


<details open>
<summary><b>Deformable DETR</b></summary>
This is a standard code of Deformable Detr for training DOTA datasets. The complete paper code will be open-sourced later (if the paper is accepted).
</details>

### Train

```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_dota.py 2
```
### Test
```
CUDA_VISIBLE_DEVICES=5 ./tools/dist_test.sh configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_dota.py work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_dota/epoch_50.pth 1 --format-only --eval-options submission_dir=work_dirs/deformable_detr_twostage_refine_r50_16x2_50e_dota/Task1_results 
```
### Some Results
![image](https://user-images.githubusercontent.com/26215859/174222334-df51f640-c267-4f1e-a9e4-25edd2b9eee1.png)

![image](https://user-images.githubusercontent.com/26215859/174222294-68698a0b-8d82-41c0-8c02-a2aa182f8e42.png)


## Installation

Please refer to [install.md](docs/en/install.md) for installation guide.

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

