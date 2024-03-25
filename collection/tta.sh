# model={'ViT-B_16', 'ViT-L_16', 'ViT_AugReg-B_16', 'ViT_AugReg-L_16', 'resnet50', 'resnet101', 'mlpmixer_B16', 'mlpmixer_L16', 'DeiT-B', 'DeiT-S', 'Beit-B16_224', 'Beit-L16_224'}
# method={'cfa', 't3a', 'shot-im', 'tent', 'pl', 'source'}

model='ViT-B_16'
# model='resnet50'
method='cfa'

CUDA_VISIBLE_DEVICES=7 python main.py \
    --tta_flag \
    --model=${model} \
    --method=${method} \
    --specify_corruption_severity=5