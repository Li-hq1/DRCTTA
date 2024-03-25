from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from robustbench.utils import set_Convpass, Convpass
from adaptBN import AlignBatchNorm
from robustbench.model_zoo.architectures.wide_resnet import NetworkBlock

class Adapter(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss()

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        model.to(x.device)
        outputs = model(x)
        # EnMin loss
        loss_EnMin = softmax_entropy(outputs).mean(0)
        # Align loss
        # loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(x.device)
        # loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(x.device)
        # clean_mean, clean_var = [], []
        # run_mean, run_var = [], []
        # for nm, m in model.named_modules():
        #     # if isinstance(m, Convpass):
        #     #     for np, p in m.named_parameters():
        #     #         if p.requires_grad:
        #     #             print(nm, np)
        #     #             print(p)
        #     #             break
        #     if isinstance(m, AlignBatchNorm):
        #         clean_mean.append(m.layer.running_mean)
        #         clean_var.append(m.layer.running_var)
        #         run_mean.append(m.norm.running_mean)
        #         run_var.append(m.norm.running_var)

        # for i in range(len(clean_mean)):
        #     loss_mean += self.l1_loss(clean_mean[i], run_mean[i])
        # for i in range(len(clean_var)):
        #     loss_var += self.l1_loss(clean_var[i], run_var[i])

        # l2 loss
        # res, raw = [], []
        # for _ in model.children():
        #     if type(_) == NetworkBlock:
        #         for i in range(4):
        #             res.append(_.layer[i].res)
        #             raw.append(_.layer[i].raw)
        # loss_regular = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(x.device)
        # for i in range(len(res)):
        #     loss_regular += self.l2_loss(res[i], raw[i])

        # loss
        # loss = loss_EnMin + (loss_mean + loss_var) * 0.5
        # loss = (loss_mean + loss_var) * 0.5

        
        loss = loss_EnMin
        # loss = loss_EnMin + loss_regular

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs

# Entropy Minimization Loss
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
# def configure_model(model):
#     dis = []
#     """Configure model for use with tent."""
#     set_Convpass(model, dim=16, s=1, xavier_init=False)
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for nm, m in model.named_modules():
#     # for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             if 'b_norm' not in nm:
#                 dis.append(m.running_mean)
#                 dis.append(m.running_var) 
#             # m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         if 'adapter' in nm or 'b_norm' in nm:
#             m.requires_grad_(True)
#     return model, dis

# def configure_model(model):
#     """Configure model for use with tent."""
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for nm, m in model.named_modules():
#     # for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#     return model



# def configure_model(model):
#     """Configure model for use with tent."""
#     # set_Convpass(model, dim=8, s=1, xavier_init=False)
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for nm, m in model.named_modules():
#     # for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         if 'adapter' in nm:
#             m.requires_grad_(True)
#             print(nm, m)
#     return model


def configure_model(model):
    """Configure model for adaptation by test-time normalization."""
    set_Convpass(model, dim=16, xavier_init=False)
    # model = AlignBatchNorm.adapt_model(model)
    model.train()
    model.requires_grad_(False)
    
    for nm, m in model.named_modules():
        # if isinstance(m, nn.BatchNorm2d):
        #     m.requires_grad_(True)
        #     # force use of batch stats in train and eval modes
        #     m.track_running_stats = False
        #     m.running_mean = None
        #     m.running_var = None
        if isinstance(m, Convpass):
            m.requires_grad_(True)
    return model

# get params for optimized
def collect_params(model):
    """Collect the normalization stats from batch norms.
    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # if isinstance(m, nn.BatchNorm2d):
        if isinstance(m, Convpass):
            for np, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names







# tent
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
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names

# def configure_model(model):
#     """Configure model for use with tent."""
#     # set_Convpass(model, dim=8, s=1, xavier_init=False)
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for nm, m in model.named_modules():
#     # for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         # if 'adapter' in nm:
#         #     m.requires_grad_(True)
#             # print(nm, m)
#     return model