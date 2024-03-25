# Domain-Regressive Continual Test-Time Adaptation


- This is a PyTorch/GPU Implementation of the paper Domain-Regressive Continual Test-Time Adaptation with Orthogonal Low-Rank Adapters. Our code is mainly based on the [official PyTorch implementation of CoTTA](https://github.com/qinenergy/cotta). 
- We have released the code about statistical characteristics collection on ViT based on the [official PyTorch implementation of CFA](https://github.com/kojima-takeshi188/CFA)
- We are committed to releasing our code upon acceptance of our paper.

## Dependencies

System

```bash
ubuntu 20.04
python 3.9.7
cuda 11.2
```

Packages

```bash
torch==1.10.0
torchvision==0.11.
timm==0.4.12
```

Environments

```
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f envs.yml
conda activate DRCTTA
```

## Datasets

Dataset ImageNet-C can be downloaded from [here](https://zenodo.org/record/2235448#.Yj2RO_co_mF).

## DRCTTA

Domain regressive continual test-time adaptation on Transformers:

```bash
cd imagenet
bash run.sh
```

Domain regressive continual test-time adaptation on CNNs:

```bash
cd cifar
# cifar10
bash run_cifar10.sh
# cifar100
bash run_cifar100.sh
```

Collect statistical characteristics of features before LN layers of ViT:

```bash
cd statistics_collection
bash statistic.sh
```



