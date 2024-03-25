#source /cluster/home/qwang/miniconda3/etc/profile.d/conda.sh
# Clean PATH and only use cotta env
# export PYTHONPATH=
# conda deactivate
# conda activate cotta
# Source-only and AdaBN results are not affected by the order as no training is performed. Therefore only need to run once.
# CUDA_VISIBLE_DEVICES=7 python -u imagenetc.py --cfg cfgs/source.yaml 
# CUDA_VISIBLE_DEVICES=1 python -u imagenetc.py --cfg cfgs/norm.yaml
# CUDA_VISIBLE_DEVICES=7 python -u imagenetc.py --cfg cfgs/tent.yaml
# CUDA_VISIBLE_DEVICES=7 python -u imagenetc.py --cfg cfgs/cotta.yaml


# CUDA_VISIBLE_DEVICES=1 python -u cfa.py --cfg cfgs/source.yaml
# CUDA_VISIBLE_DEVICES=5 python -u cfa.py --cfg cfgs/tent.yaml
# CUDA_VISIBLE_DEVICES=6 python -u cfa.py --cfg cfgs/cfa.yaml
CUDA_VISIBLE_DEVICES=7 python -u cfa.py --cfg cfgs/adapter.yaml


# CUDA_VISIBLE_DEVICES=7 python -u cfa.py --cfg cfgs/adapter_new.yaml

# CUDA_VISIBLE_DEVICES=6 python -u my_imagenetc.py --cfg cfgs/resnet.yaml

# TENT and CoTTA results are affected by the corruption sequence order
# for i in {0..9}
# do
#     CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/tent/tent$i.yaml
#     CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/cotta/cotta$i.yaml
# done
# # Run Mean and AVG for TENT and CoTTA
# cd output
# python3 -u ../eval.py | tee result.log


# CUDA_VISIBLE_DEVICES=1 python -u cfa.py --cfg cfgs/tent.yaml
# CUDA_VISIBLE_DEVICES=1 python -u cfa.py --cfg cfgs/cfa.yaml
# CUDA_VISIBLE_DEVICES=1 python -u cfa.py --cfg cfgs/adapter.yaml