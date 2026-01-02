# FractalCloud

Official PyTorch implementation for the HPCA'26 paper:

**FractalCloud: A Fractal-Inspired Architecture for  Efficient Large-Scale Point Cloud Processing**

*by [Yuzhe Fu](https://yuzhe-fu.github.io), [Changchun Zhou](https://changchun-zhou.github.io), [Hancheng Ye](https://hanchengye.com), [Bowen Duan](https://orcid.org/0009-0004-9085-5025), [Qiyu Huang](https://orcid.org/0009-0000-1970-9894), [Chiyue Wei](https://dubcyfor3.github.io), [Cong Guo](https://guocong.me), [Hai “Helen” Li](https://ece.duke.edu/people/hai-helen-li/), [Yiran Chen](https://ece.duke.edu/people/yiran-chen/)*

[[Paper (arXiv)](https://arxiv.org/abs/2511.07665)]

<p align="center">
  <img src="./assets/FractalCloud.png" width="70%">
</p>


## Abstract

This repository provides the software implementation of HPCA'26 **FractalCloud**, which introduces a fractal partitioning algorithm to partition large point clouds into spatially coherent local blocks. Based on this, FractalCloud further introduces block-parallel point operations that decompose all-point computations into local operators. We ensure numerical consistency between the software implementation and the hardware accelerator.

> **Note:** This repository is to validate the **algorithmic correctness** of our proposed design. Therefore, we do not introduce custom GPU kernel optimizations or GPU-specific acceleration.

After installation, this repository allows reproducing the network performance reported in the paper on:

- **Classification:** ModelNet40 (PointNet++, PointNeXt-S)  
- **Segmentation:** S3DIS (PointNet++, PointNeXt-S, PointVector-L)

The figure below summarizes the reproduced model accuracies:

<p align="center">
  <img src="./assets/model_accuracy.svg" width="70%">
</p>

> **Note:** We recommend using TITAN-class, RTX6000, RTX3090, or A100 GPUs (all tested successfully). Hopper architecture GPUs (e.g., H100) are not recommended. All results in this repo were obtained using TITAN GPUs.

## Installation

Firstly, please Clone the repository:

```bash
git clone https://github.com/Yuzhe-Fu/FractalCloud.git
cd FractalCloud
```

### Environment Setup
----
We provide two environment setups: Docker (recommended) or local installation.

#### Option 1: Docker (recommended)
> Time: 20-30 min for downloading, 5-10min for one-click setup. 

We recommend downloading from HuggingFace (stable).
The archive file is approximately 45 GB.

```bash
# Download from HuggingFace (recommended)
wget https://huggingface.co/YuzheFu/FractalCloud/resolve/main/FractalCloud_docker.tar

# Or Download from google drive
gdown --fuzzy "https://drive.google.com/file/d/1bjkS6beJeIV8MLgCd0CKbMack_s5fmAt/view"
```

Import the Docker image. (Please ensure Docker is installed on your system)
```bash
docker import FractalCloud_docker.tar fractalcloud_env:base
```

Start the container:
```bash
# Please run this under ./FractalCloud
docker run --name fractalcloud \
  -it --gpus all --shm-size 32G \
  -v $(pwd):/workspace \
  fractalcloud_env:base \
  /bin/bash
```
You may see a `command not found` message in the terminal. This can be safely ignored. The container automatically activates the `openpoints` conda environment with all dependencies installed.

> Note: After the container has been created once, you can re-enter it without starting a new instance:
> ```bash
> docker exec -it fractalcloud /bin/bash
> ```

#### Option 2: Local installation
> Time: 30min-1.5h, depending on your server environment.

We recommend CUDA 11.x (tested with CUDA 11.3) as required in [PointNeXt](https://github.com/guochengqian/PointNeXt). Other CUDA versions may lead to installation and execution failures. You can verify your CUDA version by running `nvcc --version`. 
> To set up a compatible CUDA 11.3 toolchain, we recommend using Anaconda for environment management and installing CUDA via conda:
> ```bash
> conda install -y cuda=11.3.1 -c nvidia/label/cuda-11.3.1
> ```

The provided installation script is based on CUDA 11.3. If a different CUDA 11.x version is
used, please adjust the script accordingly. The script will:
- Check whether `conda` is available
- Create a dedicated conda environment (`openpoints`)
- Install PyTorch and a recommended CUDA runtime automatically

```bash
source install.sh
```
> Notes: If you encounter installation issues, please refer to the [Troubleshooting Guide](https://github.com/guochengqian/PointNeXt/issues) first. Good luck!

### Pretrained Models and Datasets
----
All commands should be executed under:
-  `./workspace` (Docker setup), or
- `./FractalCloud` (local installation)

To download **Pretrained Models** and **Datasets**, please ensure that `gdown` is available in your environment (It is already included in our Docker image and install.sh).
If `gdown` is not installed, you may install it manually via: `pip install gdown`.

To download pretrained weights:
```bash
gdown --fuzzy --folder "https://drive.google.com/drive/folders/1OOlyQGHXW8NpBIot6KYSG_NkGb3NBX-p"

```

Alternatively, models can be downloaded manually from: [Google Drive link](https://drive.google.com/drive/folders/1OOlyQGHXW8NpBIot6KYSG_NkGb3NBX-p?usp=share_link) and [HuggingFace](https://huggingface.co/YuzheFu/FractalCloud/tree/main/Pretrained_Models).

Please place downloaded checkpoints into their corresponding subfolders under `./Pretrained_Models`

> Note: We also provide the evaluation logs for all evaluated models as a reference. These logs correspond to the results reported in the paper and can be used to verify reproduced accuracies.

#### Dataset Preparation
----
All commands should be executed under:
-  `./workspace` (Docker setup), or
- `./FractalCloud` (local installation)

```bash
source download_DS.sh
```


## Experiments (Model Accuracy)

All commands should be executed under:
-  `./workspace` (Docker setup), or
- `./FractalCloud` (local installation)

Below we provide example commands for reproducing evaluation results. Classification tasks use `Overall Accuracy (OA)` as the metric, while segmentation tasks use `mIoU` as the metric.

> Note: Please do not paste as one line. 

#### ModelNet40 Classification (PointNeXt-S)
```bash
# Baseline - 93.1%
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    mode=test \
    --pretrained_path ./Pretrained_Models/PNt_CLA_original/checkpoint/modelnet40_pointnext-s_ckpt_best_9311.pth

# With Fractal - 92.4%
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    mode=test \
    --fractal_stages "1,2" \
    --fractal_th 64 \
    --pretrained_path ./Pretrained_Models/PNT_CLA_fractal/checkpoint/modelnet40_pointnext-s_ckpt_best_9238.pth \
```

#### ModelNet40 Classification (PointNet++)

**Baseline 90.8%** is referenced from Fig. 16 of the *Mesorasi* paper. [[Link](https://ieeexplore.ieee.org/document/9251968)]

```bash
# With Fractal - 90.6%
CUDA_VISIBLE_DEVICES=0 bash script/main_classification.sh \
    cfgs/modelnet40ply2048/pointnet++.yaml \
    mode=test \
    --pretrained_path ./Pretrained_Models/PN++_CLA_fractal/checkpoint/modelnet40_pointnet++_ckpt_best_9056.pth \
    --fractal_stages "0,1" \
    --fractal_th 64
```

**Mesorasi 89.9%** is referenced from Fig. 16 of the *Mesorasi* paper. [[Link](https://ieeexplore.ieee.org/document/9251968)]

**Crescent 88.8%** is referenced from Fig. 13 of the *Crescent* paper. [[Link](https://dl.acm.org/doi/10.1145/3470496.3527395)]



#### S3DIS Segmentation (PointNet++)
```bash
# Baseline - 61.6%
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
    --cfg cfgs/s3dis/pointnet++.yaml \
    mode=test \
    --pretrained_path ./Pretrained_Models/PN++_SEG_original/checkpoint/s3dis-pointnet++_ckpt_best_616.pth

# With Fractal - 61.8%
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
    --cfg cfgs/s3dis/pointnet++.yaml \
    mode=test \
    --fractal_stages "0" \
    --fractal_th 256 \
    --pretrained_path ./Pretrained_Models/PN++_SEG_fractal/checkpoint/s3dis-pointnet++_ckpt_best_618.pth
```

#### S3DIS Segmentation (PointNeXt-S)
```bash
# Baseline - 62.6%
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh \
    cfgs/s3dis/pointnext-s.yaml \
    wandb.use_wandb=False \
    mode=test \
    --pretrained_path ./Pretrained_Models/PNt_SEG_original/checkpoint/s3dis-pointnext-s_ckpt_best_626.pth

# With Fractal - 62.0%
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh \
    cfgs/s3dis/pointnext-s.yaml \
    wandb.use_wandb=False \
    mode=test \
    --fractal_stages "1,2" \
    --fractal_th 256 \
    --pretrained_path ./Pretrained_Models/PNt_SEG_fractal/checkpoint/s3dis-pointnext-s_ckpt_best-620.pth
```
**PNNPU 53.8%** is referenced from Table II of the *TCAS-II* paper. [[Link](https://ieeexplore.ieee.org/document/10430381?denied=)]

#### S3DIS Segmentation (PointVector-L)
```bash
# Baseline - 70.8%
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
    --cfg cfgs/s3dis/pointvector-l.yaml \
    mode=test \
    --pretrained_path ./Pretrained_Models/PVr_SEG_original/checkpoint/s3dis-pointvector-l_ckpt_best_708.pth

# With Fractal - 70.3%
CUDA_VISIBLE_DEVICES=0 python examples/segmentation/main.py \
    --cfg cfgs/s3dis/pointvector-l.yaml \
    mode=test \
    --fractal_stages "1" \
    --fractal_th 256 \
    --pretrained_path ./Pretrained_Models/PVr_SEG_fractal/checkpoint/s3dis-pointvector-l_ckpt_best_7033.pth
```

## One more thing

### 1. Scalability

Our framework supports **both training and finetuning** for the baseline models as well as our proposed Fractal variants.  
The default mode is `training`. Setting `mode=finetune` enables finetuning from pretrained weights.

You can also use this framework to develop and train **your own partitioning method for PNNs**—simply replace the Fractal-related module with your implementation while keeping the rest of the workflow unchanged.

### Examples

```bash
# Example1: training baseline PointNeXt-S on ModelNet40.
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \

# Example2: finetuning baseline PointNeXt-S on ModelNet40.
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    mode=finetune \
    --pretrained_path ./Pretrained_Models/PNt_CLA_original/checkpoint/modelnet40_pointnext-s_ckpt_best_9311.pth

# Example3: finetuning Fractal PointNeXt-S on ModelNet40.
CUDA_VISIBLE_DEVICES=0 python examples/classification/main.py \
    --cfg cfgs/modelnet40ply2048/pointnext-s.yaml \
    mode=finetune \
    --fractal_stages "1,2" \
    --fractal_th 64 \
    --pretrained_path ./Pretrained_Models/PNT_CLA_fractal/checkpoint/modelnet40_pointnext-s_ckpt_best_9238.pth \
```
### 2. Frequent commands for docker usage
```bash
docker start fractalcloud               # start the container (needed before exec if it is stopped)
docker exec -it fractalcloud /bin/bash  # open an interactive shell inside the container
exit                                    # exit the container shell (container continues running)
docker stop fractalcloud                # stop the container
```

### 3. Notes 
> 1. Our provided `install.sh` is a simplified version of those from [PointNeXt](https://github.com/guochengqian/PointNeXt), with minimal dependencies tailored for FractalCloud. If you need the full functionality of the original repo (e.g., running PointTransformer), please refer to [PointNeXt](https://guochengqian.github.io/PointNeXt/).
> 2. Minor accuracy variations may occur across different GPU architectures (e.g., PN++_CLA_fractal: 90.56% on TITAN vs. 90.64% on RTX 3090). These differences stem from GPU-dependent numerical behavior and do not affect the overall conclusions. All paper results were obtained on TITAN GPUs for consistency.
> 3. The recursive algorithmic framework of the Fractal method implemented in `/openpoints/models/layers/{subsample.py, upsampling.py, group.py}` is **hardware-friendly and reusable in accelerator simulators**. To integrate this framework into your hardware simulation environment, one only needs to replace the existing CUDA wrappers with corresponding hardware simulation functions, while keeping the recursive structure unchanged.


## Citation
If you use this library, please kindly acknowledge our work:
```tex
@article{fu2025fractalcloud,
  title={FractalCloud: A Fractal-Inspired Architecture for Efficient Large-Scale Point Cloud Processing},
  author={Fu, Yuzhe and Zhou, Changchun and Ye, Hancheng and Duan, Bowen and Huang, Qiyu and Wei, Chiyue and Guo, Cong and Li, Hai and Chen, Yiran},
  journal={arXiv preprint arXiv:2511.07665},
  year={2025}
}
```

## Acknowledgment

This repository builds upon [OpenPoints](https://github.com/guochengqian/openpoints) and [PointNeXt](https://github.com/guochengqian/PointNeXt). We thank the authors for their open-source contributions.