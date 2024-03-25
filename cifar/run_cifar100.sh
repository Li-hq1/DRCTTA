#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
# export PYTHONPATH= 
# conda deactivate
# conda activate cotta 
# CUDA_VISIBLE_DEVICES=1 python cifar100c.py --cfg cfgs/cifar100/source.yaml
# CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/norm.yaml
# CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/tent.yaml
# CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/cotta.yaml

# CUDA_VISIBLE_DEVICES=6 python my_cifar100c.py --cfg cfgs/cifar100/cotta.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar100c.py --cfg cfgs/cifar100/source.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar100c.py --cfg cfgs/cifar100/norm.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar100c.py --cfg cfgs/cifar100/tent.yaml



# CUDA_VISIBLE_DEVICES=7 python my_cifar100c.py --cfg cfgs/cifar100/adapter.yaml
CUDA_VISIBLE_DEVICES=7 python my_cifar100c.py --cfg cfgs/cifar100/resnet_svd.yaml

# CUDA_VISIBLE_DEVICES=6 python source_dep_cifar100c.py --cfg cfgs/cifar100/tent.yaml