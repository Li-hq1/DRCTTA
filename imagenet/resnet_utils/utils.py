import timm
import torch
import math
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from robustbench.model_zoo.architectures.wide_resnet import NetworkBlock, BasicBlock

def h_std(h):
    h_hat = nn.Tanh()(nn.LayerNorm(h.shape[-1], eps=1e-05, elementwise_affine=False)(h))
    return h_hat


class MyBN(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


# adapter
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class Convpass(nn.Module):
    def __init__(self, in_dim, out_dim, stride, dim, xavier_init=False):
        super().__init__()
        self.adapter_down = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)
        self.adapter_up = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.adapter_conv = nn.Conv2d(dim, dim, 3, stride, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        self.out_dim = out_dim
        self.stride = stride

    def forward(self, x):
        # down
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv   [B C H W]
        x_down = self.act(x_down)
        # conv
        x_patch = self.adapter_conv(x_down)
        x_down = self.act(x_patch)
        # self.dropout = nn.Dropout(0.1)
        # up
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv [B C H W]
        return x_up




# Adapter
def forward_block_adapter(self, x):
        identity = x

        out = self.conv1(x)
        x_norm1 = self.bn1(out)
        out = self.relu(x_norm1)
        out = self.conv2(out)

        self.raw = out
        # print(out)
        # print(self.adapter1(x_norm1))
        out = out + self.adapter1(x_norm1)
        self.res = out

        x_norm2 = self.bn2(out)
        out = self.relu(x_norm2)
        out = self.conv3(out)

        self.raw2 = out
        out = out + self.adapter2(x_norm2)
        self.res2 = out

        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def new_forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return self.fc(x), x

# def forward(self, x):
#     return self._forward_impl(x)



def set_Convpass(model, dim=64, xavier_init=True):
    for _ in model.modules():
        if type(_) == torchvision.models.resnet.ResNet:
            bound_method = new_forward_impl.__get__(_, model.__class__)
            setattr(_, 'forward', bound_method)
    n_net = 0
    for _ in model.modules():
        if type(_) == torchvision.models.resnet.Bottleneck:
            width = 64
            planes = 64 * _.expansion
            stride = _.stride
            _.adapter1 = Convpass(width, width, stride=stride, dim=dim, xavier_init=xavier_init)
            _.adapter2 = Convpass(width, out_dim=planes, stride=1, dim=dim, xavier_init=xavier_init)
            bound_method = forward_block_adapter.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)

            n_net += 1
            if n_net >= 3:
                break

