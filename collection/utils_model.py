import os
import math
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import subprocess
import random
import timm
import datetime

def construct_model(args):
  
  if "resnet" in args.model:
    
    if "resnet50" in args.model:
        model = torchvision.models.resnet50(pretrained=True).to(args.device)
    elif "resnet101" in args.model:
        model = torchvision.models.resnet101(pretrained=True).to(args.device)
    else:
        raise ValueError("resnet model construction error !!!")
  
  # ViT original models ...
  elif ("ViT" in args.model) and ("ViT_AugReg" not in args.model):
    
    # if str(timm.__version__) != "0.4.9":
    #     print("pip uninstall timm")
    #     print("pip install timm==0.4.9")
    #     print("If \' The file might be corrupted \' error happens, \
    #            cd ~/.cache/torch/hub/checkpoints/ and rm the file and try this script again ...")
    #     raise ValueError("timm version should be 0.4.9 for ViT_Origin !!!")
        
    if "ViT-B_16" == args.model:
        model = timm.create_model('vit_base_patch16_224', pretrained=True).to(args.device)
    elif "ViT-L_16" == args.model:
        model = timm.create_model('vit_large_patch16_224', pretrained=True).to(args.device)
    else:
        raise ValueError("ViT_Origin model construction error !!!")
        
  else:
    
    if str(timm.__version__) != "0.5.0":
        print("pip uninstall timm")
        print("pip install git+https://github.com/rwightman/pytorch-image-models")
        raise ValueError("timm version should be 0.5.0 !!!")
    
    # https://github.com/rwightman/pytorch-image-models/
    # https://paperswithcode.com/lib/timm
    if "ViT_AugReg-B_16" == args.model:
        model = timm.create_model('vit_base_patch16_224', pretrained=True).to(args.device)
    elif "ViT_AugReg-L_16" == args.model:
        model = timm.create_model('vit_large_patch16_224', pretrained=True).to(args.device)
    elif "mlpmixer_B16" == args.model:
        model = timm.create_model('mixer_b16_224', pretrained=True).to(args.device)
    elif "mlpmixer_L16" == args.model:
        model = timm.create_model('mixer_l16_224', pretrained=True).to(args.device)
    elif "DeiT-B" == args.model:
        model = timm.create_model('deit_base_distilled_patch16_224', pretrained=True).to(args.device)
    elif "DeiT-S" == args.model:
        model = timm.create_model('deit_small_distilled_patch16_224', pretrained=True).to(args.device)
    elif "Beit-B16_224" == args.model:
        model = timm.create_model('beit_base_patch16_224', pretrained=True).to(args.device)
    elif "Beit-L16_224" == args.model:
        model = timm.create_model('beit_large_patch16_224', pretrained=True).to(args.device)
    else:
        raise ValueError("model construction error(model selection) !!!")
  
  def hook_fn(m, input):
        global bridging_variables
        bridging_variables = input[0]
  
  class Model_Wrapper(nn.Module):
        def __init__(self, model):
            super(Model_Wrapper, self).__init__()
            self.model = model
            if "resnet" in args.model:
                self.classifier = model.fc
                self.model.fc.register_forward_pre_hook(hook_fn)
            else:
                self.classifier = model.head
                self.model.head.register_forward_pre_hook(hook_fn)
        def forward(self, x):
            logits = self.model(x)
            h = bridging_variables
            return logits, h
  
  model = Model_Wrapper(model)
  return model






def forward_block(self, x):
    x_input = x[-1] # fix
    x_mid = x_input + self.drop_path(self.attn(self.norm1(x_input)))
    x_end = x_mid + self.drop_path(self.mlp(self.norm2(x_mid)))
    return x + (x_mid, x_end, ) # fix

def forward_features(self, x):
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    if self.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    x = (x, ) # add
    x = self.blocks(x)
    out = x[-1] # add
    out = self.norm(out)
    if self.dist_token is None:
        return self.pre_logits(out[:, 0]), x
    else:
        return out[:, 0], out[:, 1]

def forward(self, x):
    x, rec = self.forward_features(x) # fix
    if self.head_dist is not None:
        x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        if self.training and not torch.jit.is_scripting():
            # during inference, return the average of both classifier predictions
            return x, x_dist
        else:
            return (x + x_dist) / 2
    else:
        x = self.head(x)
    return x, rec # fix


# last layer
# def forward_block(self, x):
#     x_input = x
#     x_mid = x_input + self.drop_path(self.attn(self.norm1(x_input)))
#     x_end = x_mid + self.drop_path(self.mlp(self.norm2(x_mid)))
#     return x_end

# def forward_features(self, x):
#     x = self.patch_embed(x)
#     cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#     if self.dist_token is None:
#         x = torch.cat((cls_token, x), dim=1)
#     else:
#         x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#     x = self.pos_drop(x + self.pos_embed)
#     x = self.blocks(x)
#     out = x
#     out = self.norm(out)
#     if self.dist_token is None:
#         return self.pre_logits(out[:, 0])
#     else:
#         return out[:, 0], out[:, 1]

# def forward(self, x):
#     feat = self.forward_features(x)
#     if self.head_dist is not None:
#         x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
#         if self.training and not torch.jit.is_scripting():
#             # during inference, return the average of both classifier predictions
#             return x, x_dist
#         else:
#             return (x + x_dist) / 2
#     else:
#         x = self.head(feat)
#     return x, feat



def construct_model_layer_statistics(args):
    model = timm.create_model('vit_base_patch16_224', pretrained=True).to(args.device)

    bound_method = forward_features.__get__(model, model.__class__)
    setattr(model, 'forward_features', bound_method)
    bound_method = forward.__get__(model, model.__class__)
    setattr(model, 'forward', bound_method)
    for _ in model.modules():
        if type(_) == timm.models.vision_transformer.Block:
            bound_method = forward_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)

    return model




# cifar10 model
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.utils import load_model
from robustbench.model_zoo.architectures.wide_resnet import BasicBlock

# after conv
# def forward_WRN(self, x):
#     out = self.conv1(x)
#     rec = (out, )
#     rec = self.block1(rec)
#     rec = self.block2(rec)
#     rec = self.block3(rec)
#     out = rec[-1]
#     out = self.relu(self.bn1(out))
#     out = F.avg_pool2d(out, 8)
#     out = out.view(-1, self.nChannels)
#     return self.fc(out), rec


# def forward_WRN_block(self, input):
#     x = input[-1]
#     x_norm1 = self.bn1(x)
#     if not self.equalInOut:
#         x = self.relu1(x_norm1)
#     else:
#         out = self.relu1(x_norm1)
#     mid = self.conv1(out if self.equalInOut else x)


#     x_norm2 = self.bn2(mid)
#     mid = self.relu2(x_norm2)
#     if self.droprate > 0:
#         mid = F.dropout(mid, p=self.droprate, training=self.training)
#     mid = self.conv2(mid)
#     if self.equalInOut:
#         x_add = x
#     else:
#         x_add = self.convShortcut(x)
#     end = torch.add(x_add, mid)

#     return input + (mid, end, )


# after bn
# def forward_WRN(self, x):
#     out = self.conv1(x)
#     rec = [out, ]
#     rec = self.block1((out, rec))
#     rec = self.block2(rec)
#     out, rec = self.block3(rec)
#     out = self.relu(self.bn1(out))
#     out = F.avg_pool2d(out, 8)
#     out = out.view(-1, self.nChannels)
#     return self.fc(out), rec


# def forward_WRN_block(self, input):
#     x, rec = input
#     x_norm1 = self.bn1(x)
#     if not self.equalInOut:
#         x = self.relu1(x_norm1)
#     else:
#         out = self.relu1(x_norm1)
#     mid = self.conv1(out if self.equalInOut else x)


#     x_norm2 = self.bn2(mid)
#     mid = self.relu2(x_norm2)
#     if self.droprate > 0:
#         mid = F.dropout(mid, p=self.droprate, training=self.training)
#     mid = self.conv2(mid)
#     if self.equalInOut:
#         x_add = x
#     else:
#         x_add = self.convShortcut(x)
#     end = torch.add(x_add, mid)

#     rec.append(x_norm1)
#     rec.append(x_norm2)
#     return (end, rec)


# last layer
# def forward_WRN(self, x):
#     out = self.conv1(x)
#     out = self.block1(out)
#     out = self.block2(out)
#     out = self.block3(out)
#     out = self.relu(self.bn1(out))
#     out = F.avg_pool2d(out, 8)
#     out = out.view(-1, self.nChannels)
#     return self.fc(out), out

def forward_WRN(self, x):
    out = self.conv1(x)
    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    
    # print(out.shape) # [100 640 8 8]
    feat = F.avg_pool2d(out, 8)
    B = feat.shape[0]
    feat = feat.reshape(B, -1)
    # print(feat.shape)

    out = self.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, self.nChannels)
    return self.fc(out), feat


def forward_WRN_block(self, input):
    x = input
    x_norm1 = self.bn1(x)
    if not self.equalInOut:
        x = self.relu1(x_norm1)
    else:
        out = self.relu1(x_norm1)
    mid = self.conv1(out if self.equalInOut else x)


    x_norm2 = self.bn2(mid)
    mid = self.relu2(x_norm2)
    if self.droprate > 0:
        mid = F.dropout(mid, p=self.droprate, training=self.training)
    mid = self.conv2(mid)
    if self.equalInOut:
        x_add = x
    else:
        x_add = self.convShortcut(x)
    end = torch.add(x_add, mid)

    return end


def construct_model_cifar10(args):
    model = load_model('Standard', 'ckpt',
                       BenchmarkDataset.cifar_10, ThreatModel.corruptions).to(args.device)
    bound_method = forward_WRN.__get__(model, model.__class__)
    setattr(model, 'forward', bound_method)
    for _ in model.modules():
        if type(_) == BasicBlock:
            bound_method = forward_WRN_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
    return model


def construct_model_cifar100(args):
    model = load_model('Hendrycks2020AugMix_WRN', 'ckpt',
                       BenchmarkDataset.cifar_100, ThreatModel.corruptions).to(args.device)
    bound_method = forward_WRN.__get__(model, model.__class__)
    setattr(model, 'forward', bound_method)
    for _ in model.modules():
        if type(_) == BasicBlock:
            bound_method = forward_WRN_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
    return model



def load_model_from_saved_file(args):
  model = construct_model(args)
#   model = construct_model_layer_statistics(args)
  model.load_state_dict(torch.load(args.model_save_file)) #, strict=False)
  print("The model is restored ...")
  return model

def save_model_to_file(model, file_path):
  torch.save(model.state_dict(), file_path)
  print("The model is saved ...")
