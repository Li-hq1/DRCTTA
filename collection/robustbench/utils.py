import argparse
import dataclasses
import json
import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union
from einops import rearrange, repeat

import requests
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.model_zoo.architectures.wide_resnet import NetworkBlock, BasicBlock
from torch.nn.modules.batchnorm import _BatchNorm

ACC_FIELDS = {
    ThreatModel.corruptions: "corruptions_acc",
    ThreatModel.L2: "autoattack_acc",
    ThreatModel.Linf: "autoattack_acc"
}


def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def add_substr_to_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[substr + k] = v
    return new_state_dict


def load_model(model_name: str,
               model_dir: Union[str, Path] = './models',
               dataset: Union[str,
                              BenchmarkDataset] = BenchmarkDataset.cifar_10,
               threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
               norm: Optional[str] = None) -> nn.Module:
    """Loads a model from the model_zoo.

     The model is trained on the given ``dataset``, for the given ``threat_model``.

    :param model_name: The name used in the model zoo.
    :param model_dir: The base directory where the models are saved.
    :param dataset: The dataset on which the model is trained.
    :param threat_model: The threat model for which the model is trained.
    :param norm: Deprecated argument that can be used in place of ``threat_model``. If specified, it
      overrides ``threat_model``

    :return: A ready-to-used trained model.
    """

    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    model_dir_ = Path(model_dir) / dataset_.value / threat_model_.value
    model_path = model_dir_ / f'{model_name}.pt'

    models = all_models[dataset_][threat_model_]

    if not isinstance(models[model_name]['gdrive_id'], list):
        model = models[model_name]['model']()
        if dataset_ == BenchmarkDataset.imagenet and 'Standard' in model_name:
            return model.eval()
        
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        if not os.path.isfile(model_path):
            download_gdrive(models[model_name]['gdrive_id'], model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if 'Kireev2021Effectiveness' in model_name or model_name == 'Andriushchenko2020Understanding':
            checkpoint = checkpoint['last']  # we take the last model (choices: 'last', 'best')
        try:
            # needed for the model of `Carmon2019Unlabeled`
            state_dict = rm_substr_from_state_dict(checkpoint['state_dict'],
                                                   'module.')
            # needed for the model of `Chen2020Efficient`
            state_dict = rm_substr_from_state_dict(state_dict,
                                                   'model.')
        except:
            state_dict = rm_substr_from_state_dict(checkpoint, 'module.')
            state_dict = rm_substr_from_state_dict(state_dict, 'model.')

        if dataset_ == BenchmarkDataset.imagenet:
            # so far all models need input normalization, which is added as extra layer
            state_dict = add_substr_to_state_dict(state_dict, 'model.')
        
        model = _safe_load_state_dict(model, model_name, state_dict, dataset_)

        return model.eval()

    # If we have an ensemble of models (e.g., Chen2020Adversarial)
    else:
        model = models[model_name]['model']()
        if not os.path.exists(model_dir_):
            os.makedirs(model_dir_)
        for i, gid in enumerate(models[model_name]['gdrive_id']):
            if not os.path.isfile('{}_m{}.pt'.format(model_path, i)):
                download_gdrive(gid, '{}_m{}.pt'.format(model_path, i))
            checkpoint = torch.load('{}_m{}.pt'.format(model_path, i),
                                    map_location=torch.device('cpu'))
            try:
                state_dict = rm_substr_from_state_dict(
                    checkpoint['state_dict'], 'module.')
            except KeyError:
                state_dict = rm_substr_from_state_dict(checkpoint, 'module.')

            model.models[i] = _safe_load_state_dict(model.models[i],
                                                    model_name, state_dict,
                                                    dataset_)
            model.models[i].eval()

        return model.eval()


def _safe_load_state_dict(model: nn.Module, model_name: str,
                          state_dict: Dict[str, torch.Tensor],
                          dataset_: BenchmarkDataset) -> nn.Module:
    known_failing_models = {
        "Andriushchenko2020Understanding", "Augustin2020Adversarial",
        "Engstrom2019Robustness", "Pang2020Boosting", "Rice2020Overfitting",
        "Rony2019Decoupling", "Wong2020Fast", "Hendrycks2020AugMix_WRN",
        "Hendrycks2020AugMix_ResNeXt", "Kireev2021Effectiveness_Gauss50percent",
        "Kireev2021Effectiveness_AugMixNoJSD", "Kireev2021Effectiveness_RLAT",
        "Kireev2021Effectiveness_RLATAugMixNoJSD", "Kireev2021Effectiveness_RLATAugMixNoJSD",
        "Kireev2021Effectiveness_RLATAugMix", "Chen2020Efficient",
        "Wu2020Adversarial", "Augustin2020Adversarial_34_10",
        "Augustin2020Adversarial_34_10_extra"
    }

    failure_messages = ['Missing key(s) in state_dict: "mu", "sigma".',
                        'Unexpected key(s) in state_dict: "model_preact_hl1.1.weight"',
                        'Missing key(s) in state_dict: "normalize.mean", "normalize.std"']

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if (model_name in known_failing_models or dataset_ == BenchmarkDataset.imagenet
            ) and any([msg in str(e) for msg in failure_messages]):
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e

    return model


def clean_accuracy(i_c, 
                   log_writer, 
                   model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
            # tensorboard
            cur_acc = (output.max(1)[1] == y_curr).float().sum() / output.shape[0]
            log_writer.add_scalar('acc', cur_acc, i_c * n_batches + counter)
            log_writer.flush()
    return acc.item() / x.shape[0]


def list_available_models(
        dataset: Union[str, BenchmarkDataset] = BenchmarkDataset.cifar_10,
        threat_model: Union[str, ThreatModel] = ThreatModel.Linf,
        norm: Optional[str] = None):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)

    if norm is None:
        threat_model_: ThreatModel = ThreatModel(threat_model)
    else:
        threat_model_ = ThreatModel(norm)
        warnings.warn(
            "`norm` has been deprecated and will be removed in a future version.",
            DeprecationWarning)

    models = all_models[dataset_][threat_model_].keys()

    acc_field = ACC_FIELDS[threat_model_]

    json_dicts = []

    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value

    for model_name in models:
        json_path = jsons_dir / f"{model_name}.json"

        # Some models might not yet be in model_info
        if not json_path.exists():
            continue

        with open(json_path, 'r') as model_info:
            json_dict = json.load(model_info)

        json_dict['model_name'] = model_name
        json_dict['venue'] = 'Unpublished' if json_dict[
            'venue'] == '' else json_dict['venue']
        json_dict[acc_field] = float(json_dict[acc_field]) / 100
        json_dict['clean_acc'] = float(json_dict['clean_acc']) / 100
        json_dicts.append(json_dict)

    json_dicts = sorted(json_dicts, key=lambda d: -d[acc_field])
    print('| <sub>#</sub> | <sub>Model ID</sub> | <sub>Paper</sub> | <sub>Clean accuracy</sub> | <sub>Robust accuracy</sub> | <sub>Architecture</sub> | <sub>Venue</sub> |')
    print('|:---:|---|---|:---:|:---:|:---:|:---:|')
    for i, json_dict in enumerate(json_dicts):
        if json_dict['model_name'] == 'Chen2020Adversarial':
            json_dict['architecture'] = json_dict[
                'architecture'] + ' <br/> (3x ensemble)'
        if json_dict['model_name'] != 'Natural':
            print(
                '| <sub>**{}**</sub> | <sub><sup>**{}**</sup></sub> | <sub>*[{}]({})*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'
                .format(i + 1, json_dict['model_name'], json_dict['name'],
                        json_dict['link'], json_dict['clean_acc'],
                        json_dict[acc_field], json_dict['architecture'],
                        json_dict['venue']))
        else:
            print(
                '| <sub>**{}**</sub> | <sub><sup>**{}**</sup></sub> | <sub>*{}*</sub> | <sub>{:.2%}</sub> | <sub>{:.2%}</sub> | <sub>{}</sub> | <sub>{}</sub> |'
                .format(i + 1, json_dict['model_name'], json_dict['name'],
                        json_dict['clean_acc'], json_dict[acc_field],
                        json_dict['architecture'], json_dict['venue']))


def _get_bibtex_entry(model_name: str, title: str, authors: str, venue: str, year: int):
    authors = authors.replace(', ', ' and ')
    return (f"@article{{{model_name},\n"
            f"\ttitle\t= {{{title}}},\n"
            f"\tauthor\t= {{{authors}}},\n"
            f"\tjournal\t= {{{venue}}},\n"
            f"\tyear\t= {{{year}}}\n"
            "}\n")


def get_leaderboard_bibtex(dataset: Union[str, BenchmarkDataset], threat_model: Union[str, ThreatModel]):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value

    bibtex_entries = set()

    for json_path in jsons_dir.glob("*.json"):

        model_name = json_path.stem.split("_")[0]

        with open(json_path, 'r') as model_info:
            model_dict = json.load(model_info)
            title = model_dict["name"]
            authors = model_dict["authors"]
            full_venue = model_dict["venue"]
            if full_venue == "N/A":
                continue
            venue = full_venue.split(" ")[0]
            venue = venue.split(",")[0]

            year = model_dict["venue"].split(" ")[-1]

            bibtex_entry = _get_bibtex_entry(
                model_name, title, authors, venue, year)
            bibtex_entries.add(bibtex_entry)

    str_entries = ''
    for entry in bibtex_entries:
        print(entry)
        str_entries += entry

    return bibtex_entries, str_entries


def get_leaderboard_latex(dataset: Union[str, BenchmarkDataset],
                          threat_model: Union[str, ThreatModel],
                          l_keys=['clean_acc', 'external', #'autoattack_acc',
                                  'additional_data', 'architecture', 'venue',
                                  'modelzoo_id'],
                          sort_by='external' #'autoattack_acc'
                          ):
    dataset_: BenchmarkDataset = BenchmarkDataset(dataset)
    threat_model_: ThreatModel = ThreatModel(threat_model)

    models = all_models[dataset_][threat_model_]
    print(models.keys())
    
    jsons_dir = Path("./model_info") / dataset_.value / threat_model_.value
    entries = []

    for json_path in jsons_dir.glob("*.json"):
        if not json_path.stem.startswith('Standard'):
            model_name = json_path.stem.split("_")[0]
        else:
            model_name = json_path.stem
        
        with open(json_path, 'r') as model_info:
            model_dict = json.load(model_info)

        str_curr = '\\citet{{{}}}'.format(model_name) if not model_name in ['Standard', 'Standard_R50'] \
            else model_name.replace('_', '\\_')

        for k in l_keys:
            if k == 'external' and not 'external' in model_dict.keys():
                model_dict[k] = model_dict['autoattack_acc']
            if k == 'additional_data':
                v = 'Y' if model_dict[k] else 'N'
            elif k == 'architecture':
                v = model_dict[k].replace('WideResNet', 'WRN')
                v = v.replace('ResNet', 'RN')
            elif k == 'modelzoo_id':
                # print(json_path.stem)
                v = json_path.stem.split('.json')[0]
                if not v in models.keys():
                    v = 'N/A'
                else:
                    v = v.replace('_', '\\_')
            else:
                v = model_dict[k]
            str_curr += ' & {}'.format(v)
        str_curr += '\\\\'
        entries.append((str_curr, float(model_dict[sort_by])))

    entries = sorted(entries, key=lambda k: k[1], reverse=True)
    entries = ['{} &'.format(i + 1) + a for i, (a, b) in enumerate(entries)]
    entries = '\n'.join(entries).replace('<br>', ' ')

    return entries


def update_json(dataset: BenchmarkDataset, threat_model: ThreatModel,
                model_name: str, accuracy: float, adv_accuracy: float,
                eps: Optional[float]) -> None:
    json_path = Path(
        "model_info"
    ) / dataset.value / threat_model.value / f"{model_name}.json"
    if not json_path.parent.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)

    acc_field = ACC_FIELDS[threat_model]

    acc_field_kwarg = {acc_field: adv_accuracy}

    model_info = ModelInfo(dataset=dataset.value, eps=eps, clean_acc=accuracy, **acc_field_kwarg)

    with open(json_path, "w") as f:
        f.write(json.dumps(dataclasses.asdict(model_info), indent=2))


@dataclasses.dataclass
class ModelInfo:
    link: Optional[str] = None
    name: Optional[str] = None
    authors: Optional[str] = None
    additional_data: Optional[bool] = None
    number_forward_passes: Optional[int] = None
    dataset: Optional[str] = None
    venue: Optional[str] = None
    architecture: Optional[str] = None
    eps: Optional[float] = None
    clean_acc: Optional[float] = None
    reported: Optional[float] = None
    corruptions_acc: Optional[str] = None
    autoattack_acc: Optional[str] = None
    footnote: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        choices=[x.value for x in ThreatModel])
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar10',
                        choices=[x.value for x in BenchmarkDataset])
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data',
                        help='where to store downloaded datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='./models',
                        help='where to store downloaded models')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    parser.add_argument('--to_disk', type=bool, default=True)
    args = parser.parse_args()
    return args

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        # if (not self.training) or (np.random.random()) > self.p:
        #     return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        return x


class LDP(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-5):
        super(LDP, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

        # self.gamma = nn.Parameter(torch.ones(dim))
        # self.beta = nn.Parameter(torch.zeros(dim))
        self.lock = False
        self.decay = 0.9


        self.count = 0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):

        N,C,H,W = x.size()
        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        if not self.lock:
            self.gamma = std
            self.beta = mean
        else:
            self.gamma = self.decay * self.gamma + (1 - self.decay) * std
            self.beta = self.decay * self.beta + (1 - self.decay) * mean
            self.lock = True

        # sqrtvar_mu = self.sqrtvar(self.beta)
        # sqrtvar_std = self.sqrtvar(self.gamma)
        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        return x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           







# forward replacement of layer in conv-block 
# def forward_block(self, x):

#     if not self.equalInOut:
#         x = self.relu1(self.bn1(x))
#     else:
#         out = self.relu1(self.bn1(x))
#     out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
#     if self.droprate > 0:
#         out = F.dropout(out, p=self.droprate, training=self.training)
#     out = self.conv2(out)
#     if self.equalInOut:
#         out = torch.add(x, out)
#     else:
#         out = torch.add(self.convShortcut(x), out)
        
#     self.raw = out
#     out = out + self.adapter(self.bn1(x))
#     self.res = out
#     # out = out + self.adapter(x)
#     return out



def forward_align_block(self, x):

    if not self.equalInOut:
        x = self.relu1(self.bn1(x))
    else:
        out = self.relu1(self.bn1(x))
    out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
    if self.droprate > 0:
        out = F.dropout(out, p=self.droprate, training=self.training)
    out = self.conv2(out)
    if self.equalInOut:
        out = torch.add(x, out)
    else:
        out = torch.add(self.convShortcut(x), out)
    
    
    out = self.mean_align(out)
    self.raw = out
    out = out + self.adapter(self.bn1(x))
    self.res = out
    # out = out + self.adapter(x)
    return out


# adapter
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# class Convpass(nn.Module):
#     def __init__(self, in_dim, out_dim, stride, dim, xavier_init=False):
#         super().__init__()
#         self.adapter_conv = nn.Conv2d(dim, dim, 3, stride, 1)
#         if xavier_init:
#             nn.init.xavier_uniform_(self.adapter_conv.weight)
#         else:
#             nn.init.zeros_(self.adapter_conv.weight)
#             self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
#         nn.init.zeros_(self.adapter_conv.bias)

#         self.adapter_down = nn.Linear(in_dim, dim)  # equivalent to 1 * 1 Conv
#         self.adapter_up = nn.Linear(dim, out_dim)  # equivalent to 1 * 1 Conv
#         nn.init.xavier_uniform_(self.adapter_down.weight)
#         # nn.init.zeros_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.bias)
#         nn.init.zeros_(self.adapter_up.weight)
#         nn.init.zeros_(self.adapter_up.bias)

#         self.act = QuickGELU()
#         self.norm = nn.LayerNorm(dim)
#         # self.act = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#         self.out_dim = out_dim
#         self.stride = stride

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, H*W).permute(0, 2, 1) #B, N, C

#         x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
#         # x_down = x_down.permute(0, 2, 1)
#         # x_down = F.layer_norm(x_down, [H*W])
#         x_down = self.act(x_down)
#         # x_down = x_down.permute(0, 2, 1)
#         # x_down = self.act(x_down)

#         x_patch = x_down.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
#         x_patch = self.adapter_conv(x_patch)
#         x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, -1, self.dim)

#         x_down = self.act(x_patch)
#         # x_down = self.dropout(x_down)
#         x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
#         x_up = x_up.reshape(B, H // self.stride, W // self.stride, self.out_dim).permute(0, 3, 1, 2)

#         return x_up



# class Convpass(nn.Module):
#     def __init__(self, in_dim, out_dim, stride, dim, xavier_init=False):
#         super().__init__()
#         # self.atten_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
#         # nn.init.xavier_uniform_(self.atten_conv.weight)
#         # nn.init.zeros_(self.atten_conv.bias)
#         # self.atten_proj = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)

#         self.adapter_down = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)
#         self.adapter_up = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0)
#         nn.init.xavier_uniform_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.bias)
#         nn.init.zeros_(self.adapter_up.weight)
#         nn.init.zeros_(self.adapter_up.bias)

#         self.adapter_conv = nn.Conv2d(dim, dim, 3, stride, 1)
#         if xavier_init:
#             nn.init.xavier_uniform_(self.adapter_conv.weight)
#         else:
#             nn.init.zeros_(self.adapter_conv.weight)
#             self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
#         nn.init.zeros_(self.adapter_conv.bias)

#         self.act = QuickGELU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#         self.out_dim = out_dim
#         self.stride = stride

#     def forward(self, x):
#         # x_atten = self.atten_conv(x)
#         # x_atten = F.avg_pool2d(x_atten, 32) # [B C 1 1]
#         # x_atten = self.atten_proj(x_atten) # [B C 1 1]
#         # x_atten = torch.sigmoid(x_atten)

#         x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv   [B C H W]
#         x_down = self.act(x_down)

#         x_patch = self.adapter_conv(x_down)
#         x_down = self.act(x_patch)
#         self.dropout = nn.Dropout(0.1)

#         x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv [B C H W]

#         return x_up

class MyBN(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
        
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
    

class InputAdapter(nn.Module):
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
    


# QKV adapter
# class Convpass(nn.Module):
#     def __init__(self, in_dim, out_dim, stride, dim, xavier_init=False):
#         super().__init__()
#         self.scale = (dim ** -0.5)
#         self.softmax = nn.Softmax(dim = -1)
#         self.q = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)
#         # self.k = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)
#         self.v = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
#         nn.init.zeros_(self.v.weight)
#         nn.init.zeros_(self.v.bias)

#         # self.adapter_down = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)
#         # self.adapter_up = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
#         # nn.init.xavier_uniform_(self.adapter_down.weight)
#         # nn.init.zeros_(self.adapter_down.bias)
#         # nn.init.zeros_(self.adapter_up.weight)
#         # nn.init.zeros_(self.adapter_up.bias)

#         # self.adapter_conv = nn.Conv2d(dim, dim, 3, stride, 1)
#         # if xavier_init:
#         #     nn.init.xavier_uniform_(self.adapter_conv.weight)
#         # else:
#         #     nn.init.zeros_(self.adapter_conv.weight)
#         #     self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
#         # nn.init.zeros_(self.adapter_conv.bias)

#         # self.act = QuickGELU()
#         # self.dropout = nn.Dropout(0.1)
#         # self.dim = dim
#         # self.out_dim = out_dim
#         # self.stride = stride

#     def forward(self, x):
#         B, C, H, W = x.shape
#         q = self.q(x); q = rearrange(q, 'B C H W -> B (H W) C')
#         k = q
#         # k = self.k(x); k = rearrange(k, 'B C H W -> B (H W) C')
#         atten = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [B HW HW]
#         atten = self.softmax(atten)
#         v = self.v(x); v = rearrange(v, 'B C H W -> B (H W) C') # [B HW C]
#         out = torch.matmul(atten, v) # [B HW C]
#         out = rearrange(out, 'B (H W) C -> B C H W', H=H)

#         # x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv   [B C H W]
#         # x_down = self.act(x_down)

#         # x_patch = self.adapter_conv(x_down)
#         # x_down = self.act(x_patch)
#         # self.dropout = nn.Dropout(0.1)

#         # x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv [B C H W]
#         return out
    

# class Convpass(nn.Module):
#     def __init__(self, in_dim, out_dim, stride, dim, xavier_init=False):
#         super().__init__()
#         self.adapter_down = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)
#         self.adapter_up = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0)
#         nn.init.xavier_uniform_(self.adapter_down.weight)
#         nn.init.zeros_(self.adapter_down.bias)
#         nn.init.zeros_(self.adapter_up.weight)
#         nn.init.zeros_(self.adapter_up.bias)

#         self.adapter_conv = nn.Conv2d(dim, dim, 3, stride, 1)
#         if xavier_init:
#             nn.init.xavier_uniform_(self.adapter_conv.weight)
#         else:
#             nn.init.zeros_(self.adapter_conv.weight)
#             self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
#         nn.init.zeros_(self.adapter_conv.bias)

#         self.act = QuickGELU()
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim
#         self.out_dim = out_dim
#         self.stride = stride

#     def forward(self, x):
#         # down
#         x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv   [B C H W]
#         x_down = self.act(x_down)
#         # conv
#         x_patch = self.adapter_conv(x_down)
#         x_down = self.act(x_patch)
#         self.dropout = nn.Dropout(0.1)
#         # up
#         x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv [B C H W]
#         return x_up
    

# class MeanAlign(nn.Module):
#     def __init__(self, ori_mean):
#         super().__init__()
#         self.ori_mean = ori_mean

#     def forward(self, input):
#         cur_mean = input.mean(dim=[0, 2, 3], keepdim=False)
#         delta_mean = self.ori_mean - cur_mean
#         delta_mean = delta_mean[None, :, None, None] # [1, C, 1, 1]
#         input = input + delta_mean
#         return input


# input prompt
class InputPrompter(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        # input prompt
        self.image_size = image_size
        self.prompt = nn.Parameter(torch.zeros(1, 3, self.image_size, self.image_size))

    def forward(self, x):
        return x + self.prompt
    

# WRN forward replacement
def forward_WRN(self, x):
    # x = self.prompter(x)

    # self.raw = x
    # x = x + self.adapter(x)
    # self.res = x  
    out = self.conv1(x)

    # self.raw = out
    # out = out + self.adapter(out)
    # self.res = out
    # out = out.detach()

    out = self.block1(out)
    out = self.block2(out)
    out = self.block3(out)
    out = self.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, self.nChannels)
    return self.fc(out)


def forward_block(self, x):
    x_norm1 = self.bn1(x)
    if not self.equalInOut:
        x = self.relu1(x_norm1)
    else:
        out = self.relu1(x_norm1)
    out = self.conv1(out if self.equalInOut else x)

    self.raw = out
    out = out + self.adapter1(x_norm1)
    self.res = out

    x_norm2 = self.bn2(out)
    out = self.relu2(x_norm2)
    if self.droprate > 0:
        out = F.dropout(out, p=self.droprate, training=self.training)
    out = self.conv2(out)

    self.raw2 = out
    out = out + self.adapter2(x_norm2)
    self.res2 = out

    if self.equalInOut:
        x_add = x
    else:
        x_add = self.convShortcut(x)
    out = torch.add(x_add, out)

    return out


def set_Convpass(model, dim=64, xavier_init=True):
    n_net = 0
    for net_block in model.children():
        if type(net_block) == NetworkBlock:
            for i in range(4):
                block = net_block.layer[i]
                in_dim = block.in_planes 
                out_dim = block.out_planes
                stride = block.stride
                block.adapter1 = Convpass(in_dim, out_dim, stride, dim=int(0.4*out_dim), xavier_init=xavier_init)
                block.adapter2 = Convpass(out_dim, out_dim, stride=1, dim=int(0.4*out_dim), xavier_init=xavier_init)
                bound_method = forward_block.__get__(block, block.__class__)
                setattr(block, 'forward', bound_method)
            n_net += 1
            if n_net >= 1:
                break
    
    # block = model
    # in_dim = 16
    # out_dim = 16
    # stride = 1
    # block.adapter = Convpass(in_dim, out_dim, stride, dim=32, xavier_init=xavier_init)
    # bound_method = forward_WRN.__get__(block, block.__class__)
    # setattr(block, 'forward', bound_method)

    # block = model
    # block.prompter = InputPrompter(image_size=32)


def set_Convpass100(model, dim=64, xavier_init=True):
    n_net = 0
    for _ in model.children():
        if type(_) == NetworkBlock:
            for i in range(6):
                block = _.layer[i]
                in_dim = block.in_planes
                out_dim = block.out_planes
                stride = block.stride
                block.adapter1 = Convpass(in_dim, out_dim, stride, dim=64, xavier_init=xavier_init)
                block.adapter2 = Convpass(out_dim, out_dim, stride=1, dim=64, xavier_init=xavier_init)
                bound_method = forward_block.__get__(block, block.__class__)
                setattr(block, 'forward', bound_method)
            n_net += 1
            if n_net >= 1:
                break