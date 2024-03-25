#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
# export PYTHONPATH= 
# conda deactivate
# conda activate cotta 
# CUDA_VISIBLE_DEVICES=0 python cifar10c.py --cfg cfgs/cifar10/source.yaml
# CUDA_VISIBLE_DEVICES=6 python cifar10c.py --cfg cfgs/cifar10/norm.yaml
# CUDA_VISIBLE_DEVICES=5 python cifar10c.py --cfg cfgs/cifar10/cotta.yaml




# CUDA_VISIBLE_DEVICES=6 python my_cifar10c.py --cfg cfgs/cifar10/source.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar10c.py --cfg cfgs/cifar10/norm.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar10c.py --cfg cfgs/cifar10/tent.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar10c.py --cfg cfgs/cifar10/cotta.yaml
# CUDA_VISIBLE_DEVICES=2 python my_cifar10c.py --cfg cfgs/cifar10/adapter.yaml
CUDA_VISIBLE_DEVICES=5 python my_cifar10c.py --cfg cfgs/cifar10/cnn_svd.yaml

# CUDA_VISIBLE_DEVICES=7 python my_cifar10c.py --cfg cfgs/cifar10/cotta_reverse.yaml
# CUDA_VISIBLE_DEVICES=6 python my_cifar10c.py --cfg cfgs/cifar10/adapter_reverse.yaml





# CUDA_VISIBLE_DEVICES=7 python source_dep_cifar10c.py --cfg cfgs/cifar10/adapter.yaml