
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

