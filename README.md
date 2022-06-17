
## Introduction

### AO2-DETR: Arbitrary-Oriented Object Detection Transformer (https://arxiv.org/abs/2205.12785).
![Snipaste_2022-06-17_11-58-45](https://user-images.githubusercontent.com/26215859/174222183-2de9fe00-8dd2-4535-8427-d9c385f145f8.png)


<details open>
<summary><b>Deformable DETR</b></summary>
This is a standard code of Deformable Detr for training DOTA datasets. The complete paper code will be open-sourced later (if the paper is accepted).
</details>

### Train

```   
CUDA_VISIBLE_DEVICES=2,3 ./tools/dist_train.sh configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_dota.py 2
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

