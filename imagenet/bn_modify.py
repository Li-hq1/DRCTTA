import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def set_bn(model):
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bound_method = forward.__get__(m, m.__class__)
            setattr(m, 'forward', bound_method)

def forward(self, input: Tensor) -> Tensor:
        # compute current mean and variance
        self.cur_mean = input.mean(dim=[0, 2, 3], keepdim=False)
        self.cur_var = input.var(dim=[0, 2, 3], keepdim=False)
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )





# def set_bn(model):
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             block = 1
#             m.block = block
#             m.mean_queue = m.running_mean.repeat(block, 1)
#             m.var_queue = m.running_var.repeat(block, 1)
#             m.count = 0
#             bound_method = forward.__get__(m, m.__class__)
#             setattr(m, 'forward', bound_method)

# def forward(self, input: Tensor) -> Tensor:
#         B, C, H, W = input.shape
#         # compute current mean and variance
#         self.cur_mean = input.mean(dim=[0, 2, 3], keepdim=False)
#         self.cur_var = input.var(dim=[0, 2, 3], keepdim=False)
#         self._check_input_dim(input)

#         n = B*H*W
#         # record current statistics
#         self.count = (self.count + 1) % self.block
#         mean_queue = self.mean_queue.clone()
#         var_queue = self.mean_queue.clone()
#         mean_queue[self.count] = self.cur_mean
#         var_queue[self.count] = self.cur_var
#         # update
#         self.mean_queue[self.count] = self.cur_mean.clone().detach()
#         self.var_queue[self.count] = self.cur_var.clone().detach()
#         # correct dataset statistics
#         avg_mean = torch.mean(mean_queue, dim=0, keepdim=True)
#         delta_mean = avg_mean - mean_queue # 广播机制
#         new_var = (n - 1) * var_queue + n * (delta_mean ** 2)
#         avg_var = torch.sum(new_var, dim=0) / (self.block * n - 1)
        
#         self.block_mean = avg_mean.squeeze(0)
#         self.block_var = avg_var