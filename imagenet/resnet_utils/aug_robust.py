from copy import deepcopy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from resnet_utils.utils import set_Convpass, Convpass
from bn_modify import set_bn
import numpy as np
import PIL
import torchvision
import torchvision.transforms as transforms
import my_transforms as my_transforms
from robustbench.model_zoo.architectures.wide_resnet import NetworkBlock, BasicBlock
from torchvision.models.resnet import Bottleneck

from resnet_utils.utils import h_std

def cmd_optimization(full_max_moment, h, statistics):
    loss = 0
    moments = [0] * full_max_moment
    for j in range(full_max_moment):
        if j == 0:
            moment = torch.mean(h, dim=0, keepdim=True) # [L, 1, h, q]
        else:
            moment = torch.mean(torch.pow(h - moments[0], j+1), dim=0, keepdim=True) # [L,1,h,q]
        moments[j] = moment
        loss += torch.sqrt(torch.mean(torch.square(moments[j] - statistics[j]))) / ((2 ** j))
        
    return loss

def cmd_optimization_for_cls(cls_max_moment, logits, h, statistics):

    h = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d]
    num_class = logits.shape[-1]
    
    argmax_class = F.one_hot(torch.argmax(logits, dim=1), num_classes=num_class) # [b, c] -> [b] -> [b, c]        
    argmax_class = argmax_class.clone().detach().cuda()
    argmax_class = torch.unsqueeze(argmax_class, dim=-1) # [b, c] -> [b, c, 1]
    argmax_class_num = torch.clamp(torch.sum(argmax_class, dim=0), min=1) #  [1, c, 1]
    argmax_class_idx = torch.clamp(torch.sum(argmax_class, dim=0), min=0, max=1) #  [1, c, 1]
    
    loss = 0
    moments = [0] * cls_max_moment
    for j in range(cls_max_moment):
        if j == 0:
            moment = torch.sum(argmax_class * h, dim=0, keepdim=True) / argmax_class_num # [1, c, d]
        else:
            moment = torch.sum(torch.pow(argmax_class * h - argmax_class * moments[0], j+1), \
                                dim=0, keepdim=True) / argmax_class_num
        moments[j] = moment

        loss_cls = torch.sqrt(torch.mean(torch.square(moments[j] - statistics[j]), dim=-1, keepdim=True)) # [1, c, 1]
        loss_cls = loss_cls / ((2 ** j)) # [1, c, 1]

        loss += torch.sum(loss_cls * argmax_class_idx) / torch.sum(argmax_class_idx)
    
    return loss


class CTTA(nn.Module):
    def __init__(self, model, optimizer, dis, steps=1, episodic=False):
        super().__init__()
        cm_file = '../imagenet/ckpt/cfa_model/resnet50.imagenet.imagenet_resnet_lastlayer_norm'
        # self.statistics = torch.load(cm_file, map_location='cuda')['cmd_base_mid'] # [[cur_mean, cur_var] [...] [...] x25]
        self.statistics = {}
        self.statistics['cmd_base_mid'] = torch.load(cm_file, map_location='cuda')['cmd_base_mid']
        self.statistics['cmd_base_mid_cls'] = torch.load(cm_file, map_location='cuda')['cmd_base_mid_cls']

        self.model = model
        self.optimizer = optimizer
        self.steps = steps; assert steps > 0, "requires >= 1 step(s)"
        self.episodic = episodic
        self.dis = dis

        # load warmup params
        # msg = model.load_state_dict(torch.load('output/ckpt/warmup/lr1e-2_10reg/warmup_epoch4_6.3517.pth'))
        # print(msg)

        # copy params for reset
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # loss
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss()
        # self.mse_loss = nn.MSELoss()

    def forward(self, x, y=None):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer):
        self.model.to(x.device) # 加这句好像是因为加的adapter没有转到cuda上
        # forward
        B = x.shape[0]
        logits, rec = self.model(x)
        rec = h_std(rec)

        # adapt
        # loss_tent = softmax_entropy(outputs, outputs).mean(0)
        # loss_tent = softmax_entropy(outputs).mean(0)
        loss_align = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(x.device)
        loss_reg = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(x.device)

        # l1 loss
        # show = []
        tmp = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                tmp.append((m.cur_mean, m.cur_var))
        for i in range(len(tmp)):
            # show.append((self.l1_loss(self.dis[i][0], tmp[i][0]).cpu().item(), self.l1_loss(self.dis[i][1], tmp[i][1]).cpu().item()))
            loss_mean_var = self.l1_loss(self.dis[i][0], tmp[i][0]) + self.l1_loss(self.dis[i][1], tmp[i][1])
            loss_align += 0.2 * (math.exp(2 * i/len(tmp))) * loss_mean_var


        # source dep l1 loss
        loss1 = cmd_optimization(full_max_moment=3, h=rec, statistics=self.statistics["cmd_base_mid"])
        loss2 = cmd_optimization_for_cls(cls_max_moment=1, logits=logits, h=rec, statistics=self.statistics["cmd_base_mid_cls"])
        loss_cfa = loss1 + loss2

        # l2 loss
        n_net = 0
        res, raw = [], []
        for _ in self.model.modules():
            if type(_) == torchvision.models.resnet.Bottleneck:
                res.append(_.res)
                raw.append(_.raw)
                res.append(_.res2)
                raw.append(_.raw2)
                n_net += 1
                if n_net >= 3:
                    break
        for i in range(0, len(res)):
            loss_reg += self.l2_loss(res[i], raw[i])
            
        
        # loss = loss_align + loss_reg
        # loss = loss_tent
        loss = loss_cfa
        # loss = loss_align

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return logits.detach()


# @torch.jit.script
# def softmax_entropy(x_pred, x_gt):# -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     Ent_Loss = -(x_gt.softmax(1) * x_pred.log_softmax(1)).sum(1)
#     Ent_Loss = math.exp(- 2 * Ent_Loss) * Ent_Loss
#     return Ent_Loss

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# model copy and load (reset)
def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


# configure for new adaption
def configure_model(model):
    dis = []
    set_Convpass(model)
    set_bn(model)
    model.train()  
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            dis.append((m.running_mean.clone(), m.running_var.clone()))
            m.track_running_stats = False
        if isinstance(m, Convpass):
            m.requires_grad_(True)
    return model, dis


# get params for optimized
def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, Convpass):
            for np, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names



# def configure_model(model):
#     """Configure model for use with tent."""
#     dis = []
#     set_Convpass(model)
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for n, m in model.named_modules():
#         print(n)
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#     return model, dis

# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.

#     Walk the model's modules and collect all batch normalization parameters.
#     Return the parameters and their names.

#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             # print('here')
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names