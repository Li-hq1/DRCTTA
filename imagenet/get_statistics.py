import timm 
import torch 
import torch.nn as nn


def construct_model(backbone='vit'):
    model = timm.create_model('vit_base_patch16_224', pretrained=True).cuda()

    def hook_fn(m, input):
        global bridging_variables
        bridging_variables = input[0]

    class Model_Wrapper(nn.Module):
        def __init__(self, model):
            super(Model_Wrapper, self).__init__()
            self.model = model
            if backbone == "resnet":
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


model = construct_model()
source_train_loader, source_test_loader = setup_data_loader(args, minibatch_size_target, args.dataset)

calc_cmd_statistics(args, model, source_train_loader)