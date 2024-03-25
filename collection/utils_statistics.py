import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import random
import math
import copy
import os
from utils import *
from datetime import timedelta

def calc_cmd_statistics(args, model, train_loader):
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_mid_cls_cms = 0
    mid_ent_cms = [0] * args.save_max_moment
    mid_cls_cms = [0] * args.save_max_moment
    
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, h = model(x)
        h = h_std(args, h)
        total += y.size(0)

        mid_ent_cms[0] += torch.sum(h, dim=0, keepdim=True) # [1, d] ,直接相加会替换掉原来的0值

        ####################################################
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [b] -> [b, c]
        y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [b, c] -> [b, c, 1]
        h_ext = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d] 
        mid_cls_cms[0] += torch.sum(y_onehot * h_ext, dim=0, keepdim=True) # [b, c, d] -> [1, c, d] 对应类别位置的特征将会保留，求和后得到对应类别的均值
        total_mid_cls_cms += torch.sum(y_onehot, dim=0, keepdim=True) # [1, c, 1] 对应类别的个数
        ####################################################
        # if i == 2:
        #     break
        
    mid_ent_cms[0] = copy.deepcopy(mid_ent_cms[0] / total) # [1, d]
    mid_cls_cms[0] = copy.deepcopy(mid_cls_cms[0] / total_mid_cls_cms) # [1, c, d]
    
    total = 0
    total_mid_cls_cms = 0
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, h = model(x)
        h = h_std(args, h)
        total += y.size(0)
        
        ####################################################
        y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [b] -> [b, c]
        y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [b, c] -> [b, c, 1]
        h_ext = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d]
        total_mid_cls_cms += torch.sum(y_onehot, dim=0, keepdim=True) # [1, c, 1]
        ####################################################

        for j in range(args.save_max_moment):
            if j > 0:
                mid_ent_cms[j] += torch.sum(torch.pow(h - mid_ent_cms[0], j+1), dim=0, keepdim=True)
                mid_cls_cms_add = torch.sum(torch.pow(y_onehot * h_ext - y_onehot * mid_cls_cms[0], 
                                                      j+1), dim=0, keepdim=True) # [1,c,d]
                mid_cls_cms[j] += mid_cls_cms_add # [5][1,c,d]
        # if i == 2:
        #     break
    for j in range(args.save_max_moment):
        if j > 0:
            mid_ent_cms[j] = mid_ent_cms[j] / total
            mid_cls_cms[j] = mid_cls_cms[j] / total_mid_cls_cms
    
    torch.save({'cmd_base_mid': mid_ent_cms, \
                'cmd_base_mid_cls': mid_cls_cms} , args.cm_file)


def load_cmd_statistics(args):
    statistics = {}
    statistics['cmd_base_mid'] = torch.load(args.cm_file, map_location='cuda:0')['cmd_base_mid']
    statistics['cmd_base_mid_cls'] = torch.load(args.cm_file, map_location='cuda:0')['cmd_base_mid_cls']
    print("cmd_base_mid_mean: " + str(statistics['cmd_base_mid'][0][:,:20]))
    print("cmd_base_mid_var: " + str(statistics['cmd_base_mid'][1][:,:20]))
    print("cmd_base_mid_cls_mean: " + str(statistics['cmd_base_mid_cls'][0][:, :3, :20]))
    print("cmd_base_mid_cls_var: " + str(statistics['cmd_base_mid_cls'][1][:, :3, :20]))
    
    return statistics



def calc_cmd_layer_statistics(args, model, train_loader):
  
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_mid_cls_cms = [0] * 25
    total_list = []
    total_cls_list = []
    for m in range(25):
        total_list.append([0] * args.save_max_moment)
        total_cls_list.append([0] * args.save_max_moment)
    # mid_ent_cms = [0] * args.save_max_moment
    # mid_cls_cms = [0] * args.save_max_moment
    
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        # print(len(rec))
        # print(rec[0].shape)
        # print(rec[-1].shape)
        # print(rec[0][:5])
        total += y.size(0)

        for m in range(25):
            h = rec[m][:, 0] # 只记录cls token的统计数据
            h = h_std(args, h)
            total_list[m][0] += torch.sum(h, dim=0, keepdim=True) # [1, d] ,直接相加会替换掉原来的0值

            ####################################################
            y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [b] -> [b, c]
            y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [b, c] -> [b, c, 1]
            h_ext = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d] 
            total_cls_list[m][0] += torch.sum(y_onehot * h_ext, dim=0, keepdim=True) # [b, c, d] -> [1, c, d] 对应类别位置的特征将会保留，求和后得到对应类别的均值
            total_mid_cls_cms[m] += torch.sum(y_onehot, dim=0, keepdim=True) # [1, c, 1] 对应类别的个数
            ####################################################
        
    for m in range(25):
        total_list[m][0] = copy.deepcopy(total_list[m][0] / total) # [1, d]
        total_cls_list[m][0] = copy.deepcopy(total_cls_list[m][0] / total_mid_cls_cms[m]) # [1, c, d]
        
    total = 0
    total_mid_cls_cms = [0] * 25
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(25):
            h = rec[m][:, 0]
            h = h_std(args, h)

            ####################################################
            y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [b] -> [b, c]
            y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [b, c] -> [b, c, 1]
            h_ext = torch.unsqueeze(h, dim=1) # [b, d] -> [b, 1, d]
            total_mid_cls_cms[m] += torch.sum(y_onehot, dim=0, keepdim=True) # [1, c, 1]
            ####################################################

            for j in range(args.save_max_moment):
                if j > 0:
                    total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=0, keepdim=True)
                    mid_cls_cms_add = torch.sum(torch.pow(y_onehot * h_ext - y_onehot * total_cls_list[m][0], 
                                                        j+1), dim=0, keepdim=True) # [1,c,d]
                    total_cls_list[m][j] += mid_cls_cms_add # [5][1,c,d]
        
    for m in range(25):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total
                total_cls_list[m][j] = total_cls_list[m][j] / total_mid_cls_cms[m]
    
    torch.save({'cmd_base_mid': total_list, \
                'cmd_base_mid_cls': total_cls_list} , args.cm_file)
    




# [B N D] -> [1 N D]
def calc_cmd_layer_statistics_1ND(args, model, train_loader):
  
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []


    for m in range(25):
        total_list.append([0] * args.save_max_moment)

    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(25):
            h = rec[m] # [B N D]
            h = h_std(args, h)
            total_list[m][0] += torch.sum(h, dim=0, keepdim=True) # [1 N D]

    for m in range(25):
        total_list[m][0] = copy.deepcopy(total_list[m][0] / total) # [1 N D]
        
    total = 0
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(25):
            h = rec[m] # [B N D]
            h = h_std(args, h)

            for j in range(args.save_max_moment):
                if j > 0:
                    total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=0, keepdim=True)

    for m in range(25):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total

    
    torch.save({'cmd_base_mid': total_list} , args.cm_file)


# [B N D] -> [1 1 D]
def calc_cmd_layer_statistics_11D(args, model, train_loader):
  
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []


    for m in range(25):
        total_list.append([0] * args.save_max_moment)

    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        B, N, D = rec[0].shape
        for m in range(25):
            h = rec[m] # [B N D]
            h = h_std(args, h)
            total_list[m][0] += torch.sum(h, dim=[0,1], keepdim=True) # [1 1 D]

    for m in range(25):
        total_list[m][0] = copy.deepcopy(total_list[m][0] / total / N) # [1 1 D]
        
    total = 0
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(25):
            h = rec[m] # [B N D]
            h = h_std(args, h)
            
            for j in range(args.save_max_moment):
                if j > 0:
                    total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=[0,1], keepdim=True)

    for m in range(25):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total / N
    
    torch.save({'cmd_base_mid': total_list} , args.cm_file)


# [B N D] -> [1 1 D] with cls token or mean pooling
def calc_cmd_layer_statistics_11D_clsmean(args, model, train_loader):
  model.eval()
  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []
    for m in range(25):
        total_list.append([0] * args.save_max_moment)

    # n = 2
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)
        for m in range(25):
            h = rec[m] # [B N D]
            
            if m == 0: # 注意是不带cls token玩的
                x_mean = torch.mean(h[:, 1:], dim=2, keepdim=True) # [B N-1 D] -> [B N-1 1]
                x_var = torch.var(h[:, 1:], dim=2, keepdim=True, unbiased=False) # [B N-1 D] -> [B N-1 1]
                total_list[m][0] += torch.sum(x_mean, dim=0, keepdim=True) # [1 N 1]
                total_list[m][1] += torch.sum(x_var, dim=0, keepdim=True) # [1 N 1]
            else:
                # h = h_std(args, h)
                # cls token
                # h = h[:, :1] # [B N D] -> [B 1 D]
                # mean pooling
                h = torch.mean(h[:, 1:], dim=1, keepdim=True) # [B N-1 D] -> [B 1 D]

                # pixel-wise mu sigma2
                # h = h[:, 1:] # [B N-1 D]

                # patch-wise mu sigma2
                # h = torch.mean(h[:, 1:], dim=2, keepdim=False) # [B N-1 D] -> [B N-1 (1)]

                # feature dimention mu sigma sample
                # mu = torch.mean(h[:, 1:], dim=1, keepdim=True) # [B N-1 D] -> [B 1 D]
                # sigma = torch.sqrt(torch.var(h[:, 1:], dim=1, keepdim=True, unbiased=True)) # [B N-1 D] -> [B 1 D]
                # h = torch.cat([mu, sigma], dim=2) # [B 1 2D]

                # feature dim & patch dim
                # mu = torch.mean(h[:, 1:], dim=1, keepdim=False) # [B N-1 D] -> [B (1) D]
                # mu2 = torch.mean(h[:, 1:], dim=2, keepdim=False) # [B N-1 D] -> [B N-1 (1)]
                # sigma = torch.sqrt(torch.var(h[:, 1:], dim=1, keepdim=False, unbiased=True)) # [B N-1 D] -> [B (1) D]
                # sigma2 = torch.sqrt(torch.var(h[:, 1:], dim=2, keepdim=False, unbiased=True)) # [B N-1 D] -> [B N-1 (1)]
                # # h = torch.cat([mu, mu2], dim=1) # [B D+N-1]
                # h = torch.cat([mu, sigma, mu2, sigma2], dim=1) # [B 2D+2N-2]

                # mu
                total_list[m][0] += torch.sum(h, dim=0, keepdim=True) # [1 1 D]
        # if i >= n:
        #     break

    for m in range(25):
        total_list[m][0] = copy.deepcopy(total_list[m][0] / total) # [1 1 D]
        
    total = 0
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)
        for m in range(25):
            h = rec[m] # [B N D]

            if m == 0:
                pass
            else:
                # h = h_std(args, h)
                # cls token
                # h = h[:, :1] # [B N D] -> [B 1 D]
                # mean pooling
                h = torch.mean(h[:, 1:], dim=1, keepdim=True) # [B N-1 D] -> [B 1 D]
                
                # pixel-wise mu sigma^2
                # h = h[:, 1:] # [B N-1 D]

                # patch-wise mu sigma^2
                # h = torch.mean(h[:, 1:], dim=2, keepdim=False) # [B N-1 D] -> [B N-1 (1)]

                # feature dimention mu sigma sample
                # mu = torch.mean(h[:, 1:], dim=1, keepdim=True) # [B N-1 D] -> [B 1 D]
                # sigma = torch.sqrt(torch.var(h[:, 1:], dim=1, keepdim=True, unbiased=True)) # [B N-1 D] -> [B 1 D]
                # h = torch.cat([mu, sigma], dim=2) # [B 1 2D]

                # feature dim & patch dim
                # mu = torch.mean(h[:, 1:], dim=1, keepdim=False) # [B N-1 D] -> [B (1) D]
                # mu2 = torch.mean(h[:, 1:], dim=2, keepdim=False) # [B N-1 D] -> [B N-1 (1)]
                # sigma = torch.sqrt(torch.var(h[:, 1:], dim=1, keepdim=False, unbiased=True)) # [B N-1 D] -> [B (1) D]
                # sigma2 = torch.sqrt(torch.var(h[:, 1:], dim=2, keepdim=False, unbiased=True)) # [B N-1 D] -> [B N-1 (1)]
                # # h = torch.cat([mu, mu2], dim=1) # [B D+N-1]
                # h = torch.cat([mu, sigma, mu2, sigma2], dim=1) # [B 2D+2N-2]

                # sigma^2
                for j in range(args.save_max_moment):
                    if j > 0:
                        total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=0, keepdim=True)

                # gaussian covariance matrix
                # normlized_h = (h - total_list[m][0]).squeeze(1) # [B D]
                # total_list[m][1] += (normlized_h.T @ normlized_h) # [D D]
        # if i >= n:
        #     break

    for m in range(25):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total
    
    torch.save({'cmd_base_mid': total_list} , args.cm_file)


# for each sample [B N D] -> [B N] -> 2-dim gaussian distribution
def calc_cmd_layer_statistics_patch(args, model, train_loader):
  
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []

    for m in range(25):
        total_list.append([[],[]]) # mean and variance

    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        # visualization
        # X_pos = rec[0]
        # patch_var = torch.var(X_pos, dim=2, unbiased=True) # [B N]
        # print(patch_var)
        # min_var = torch.min(patch_var[:,1:], dim=1, keepdim=True)[0] # [B 1 1]
        # print(min_var)

        # get mean and variance on each patch feature
        for m in range(25):
            h = rec[m] # [B N D]
            h = h_std(args, h)
            patch_mean = torch.mean(h, dim=2) # [B N]
            patch_var = torch.var(h, dim=2) # [B N]
            total_list[m][0].append(patch_mean.cpu())
            total_list[m][1].append(patch_var.cpu())

    # print(total_list)
    for layer_list in total_list:
        layer_list[0] = torch.cat(layer_list[0], dim=0) # [T N]
        layer_list[1] = torch.cat(layer_list[1], dim=0) # [T N]

    # 计算二维高斯变量
    feat_list = []
    for layer_list in total_list:
        mean_sample = layer_list[0].cuda().t() # [N T]
        var_sample = layer_list[1].cuda().t() # [N T]
        list_mean1 = []
        list_mean2 = []
        list_k1 = []
        list_k2 = []
        list_k3 = []
        for i in range(len(mean_sample)):
            square_matrix = torch.stack([mean_sample[i],var_sample[i]]) # [2, 256]
            rou = torch.corrcoef(square_matrix)[0, 1]
            mean1 = torch.mean(mean_sample)
            mean2 = torch.mean(var_sample)
            var1 = torch.var(mean_sample)
            var2 = torch.var(var_sample)
            p = -1 / (2*(1-rou**2))

            k1 = p / var1
            k2 = - p*2*rou / (torch.sqrt(var1) * torch.sqrt(var2))
            k3 = p / var2
            list_k1.append(k1.item())
            list_k2.append(k2.item())
            list_k3.append(k3.item())
            list_mean1.append(mean1.item())
            list_mean2.append(mean2.item())

        list_k1 = torch.tensor(list_k1).unsqueeze(1).unsqueeze(0) # [1 N 1]
        list_k2 = torch.tensor(list_k2).unsqueeze(1).unsqueeze(0) # [1 N 1]
        list_k3 = torch.tensor(list_k3).unsqueeze(1).unsqueeze(0) # [1 N 1]
        list_mean1 = torch.tensor(list_mean1).unsqueeze(1).unsqueeze(0) # [1 N 1]
        list_mean2 = torch.tensor(list_mean2).unsqueeze(1).unsqueeze(0) # [1 N 1]
        feat_list.append([list_mean1, list_mean2, list_k1, list_k2, list_k3])
        
    torch.save({'cmd_base_mid': feat_list} , args.cm_file)



# for each sample [B N D] -> [B D] -> 2-dim gaussian distribution
def calc_cmd_layer_statistics_dim(args, model, train_loader):
  
  model.eval()

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []

    for m in range(25):
        total_list.append([[],[]]) # mean and variance

    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(25):
            h = rec[m] # [B N D]
            h = h_std(args, h)
            patch_mean = torch.mean(h, dim=1) # [B D]
            patch_var = torch.var(h, dim=1) # [B D]
            total_list[m][0].append(patch_mean.cpu())
            total_list[m][1].append(patch_var.cpu())

    # print(total_list)
    for layer_list in total_list:
        layer_list[0] = torch.cat(layer_list[0], dim=0) # [T D]
        layer_list[1] = torch.cat(layer_list[1], dim=0) # [T D]

    # 计算二维高斯变量
    feat_list = []
    for layer_list in total_list:
        mean_sample = layer_list[0].cuda().t() # [D T]
        var_sample = layer_list[1].cuda().t() # [D T]
        list_mean1 = []
        list_mean2 = []
        list_k1 = []
        list_k2 = []
        list_k3 = []
        for i in range(len(mean_sample)):
            square_matrix = torch.stack([mean_sample[i],var_sample[i]]) # [2, T]
            rou = torch.corrcoef(square_matrix)[0, 1]
            mean1 = torch.mean(mean_sample)
            mean2 = torch.mean(var_sample)
            var1 = torch.var(mean_sample)
            var2 = torch.var(var_sample)
            p = -1 / (2*(1-rou**2))

            k1 = p / var1
            k2 = - p*2*rou / (torch.sqrt(var1) * torch.sqrt(var2))
            k3 = p / var2
            list_k1.append(k1.item())
            list_k2.append(k2.item())
            list_k3.append(k3.item())
            list_mean1.append(mean1.item())
            list_mean2.append(mean2.item())

        list_k1 = torch.tensor(list_k1).unsqueeze(0).unsqueeze(0) # [1 1 D]
        list_k2 = torch.tensor(list_k2).unsqueeze(0).unsqueeze(0) # [1 1 D]
        list_k3 = torch.tensor(list_k3).unsqueeze(0).unsqueeze(0) # [1 1 D]
        list_mean1 = torch.tensor(list_mean1).unsqueeze(0).unsqueeze(0) # [1 1 D]
        list_mean2 = torch.tensor(list_mean2).unsqueeze(0).unsqueeze(0) # [1 1 D]
        feat_list.append([list_mean1, list_mean2, list_k1, list_k2, list_k3])
        
    torch.save({'cmd_base_mid': feat_list} , args.cm_file)


def calc_cmd_layer_statistics_cifar10(args, model, train_loader):
  
  model.train() # 这里没有dropout，相反有bn，感觉用train mode会好一点，应该区别不大

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []
    total_mid_cls_cms = [0] * 25
    total_cls_list = []

    for m in range(25):
        total_list.append([0] * args.save_max_moment)
        total_cls_list.append([0] * args.save_max_moment)

    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)

        # print(len(rec))
        total += y.size(0)

        for m in range(25):
            h = rec[m] # [B C H W]
            # feature map
            h = h.permute(0, 2, 3, 1) # [B H W C]
            h = h_std(args, h) # 应该是在C维度上做下LN（affine=False）+tanh
            h = h.permute(0, 3, 1, 2) # [B C H W]
            total_list[m][0] += torch.sum(h, dim=0, keepdim=True) # 在B维度上做element-wise的均值 [1 C H W]
            
            ####################################################
            y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [B] -> [B c]
            y_onehot = y_onehot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(args.device) # [B c] -> [B c 1 1 1]
            # y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [B c] -> [B c 1]
            h_ext = torch.unsqueeze(h, dim=1) # [B C H W] -> [B 1 C H W] 
            total_cls_list[m][0] += torch.sum(y_onehot * h_ext, dim=0, keepdim=True) # [B c C H W] -> [1 c C H W] 对应类别位置的特征将会保留，求和后得到对应类别的均值
            total_mid_cls_cms[m] += torch.sum(y_onehot, dim=0, keepdim=True) # [1 c 1 1 1] 对应类别的个数
            ####################################################

            # pixel
            # h = h.permute(2, 3, 0, 1) # [H W B C]
            # h = nn.Tanh()(nn.LayerNorm(h.shape[-2:], eps=1e-05, elementwise_affine=False)(h)) # norm B C dim
            # h = h.permute(2, 3, 0, 1) # [B C H W]
            # total_list[m][0] += torch.sum(h, dim=(0, 1), keepdim=True) # 在B维度上做element-wise的均值 [1 1 H W]

    for m in range(25):
        total_list[m][0] = copy.deepcopy(total_list[m][0] / total) # [1 C H W]
        total_cls_list[m][0] = copy.deepcopy(total_cls_list[m][0] / total_mid_cls_cms[m]) # [1 c C H W]

    total = 0
    total_mid_cls_cms = [0] * 25
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(25):
            h = rec[m] # [B C H W]
            # feature map
            h = h.permute(0, 2, 3, 1) # [B H W C]
            h = h_std(args, h) # 应该是在C维度上做下LN（affine=False）+tanh
            h = h.permute(0, 3, 1, 2) # [B C H W]

            ####################################################
            y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [B] -> [B c]
            y_onehot = y_onehot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(args.device) # [B c] -> [B c 1 1 1]
            h_ext = torch.unsqueeze(h, dim=1) # [B C H W] -> [B 1 C H W] 
            total_mid_cls_cms[m] += torch.sum(y_onehot, dim=0, keepdim=True) # [1 c 1 1 1] 对应类别的个数
            ####################################################


            for j in range(args.save_max_moment):
                if j > 0:
                    total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=0, keepdim=True)
                    mid_cls_cms_add = torch.sum(torch.pow(y_onehot * h_ext - y_onehot * total_cls_list[m][0], j+1), dim=0, keepdim=True) # [1 c C H W]
                    total_cls_list[m][j] += mid_cls_cms_add # [5][1 c C H W]
            # pixel
            # h = h.permute(2, 3, 0, 1) # [H W B C]
            # h = nn.Tanh()(nn.LayerNorm(h.shape[-2:], eps=1e-05, elementwise_affine=False)(h)) # norm B C dim
            # h = h.permute(2, 3, 0, 1) # [B C H W]
            # for j in range(args.save_max_moment):
            #     if j > 0:
            #         total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=(0, 1), keepdim=True)

    for m in range(25):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total
                total_cls_list[m][j] = total_cls_list[m][j] / total_mid_cls_cms[m]
    
    torch.save({'cmd_base_mid': total_list} , args.cm_file)




def calc_cmd_layer_statistics_cifar100(args, model, train_loader):
  length = 37
  model.train() # 这里没有dropout，相反有bn，感觉用train mode会好一点，应该区别不大

  # No gradient during evaluation ...
  with torch.no_grad():
    total = 0
    total_list = []
    total_mid_cls_cms = [0] * length
    total_cls_list = []

    for m in range(length):
        total_list.append([0] * args.save_max_moment)
        total_cls_list.append([0] * args.save_max_moment)

    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(length):
            h = rec[m] # [B C H W]
            # feature map
            h = h.permute(0, 2, 3, 1) # [B H W C]
            h = h_std(args, h) # 应该是在C维度上做下LN（affine=False）+tanh
            h = h.permute(0, 3, 1, 2) # [B C H W]
            total_list[m][0] += torch.sum(h, dim=0, keepdim=True) # 在B维度上做element-wise的均值 [1 C H W]
            
            # ####################################################
            # y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [B] -> [B c]
            # y_onehot = y_onehot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(args.device) # [B c] -> [B c 1 1 1]
            # # y_onehot = torch.unsqueeze(y_onehot, dim=-1).to(args.device) # [B c] -> [B c 1]
            # h_ext = torch.unsqueeze(h, dim=1) # [B C H W] -> [B 1 C H W] 
            # total_cls_list[m][0] += torch.sum(y_onehot * h_ext, dim=0, keepdim=True) # [B c C H W] -> [1 c C H W] 对应类别位置的特征将会保留，求和后得到对应类别的均值
            # total_mid_cls_cms[m] += torch.sum(y_onehot, dim=0, keepdim=True) # [1 c 1 1 1] 对应类别的个数
            # ####################################################

            # pixel
            # h = h.permute(2, 3, 0, 1) # [H W B C]
            # h = nn.Tanh()(nn.LayerNorm(h.shape[-2:], eps=1e-05, elementwise_affine=False)(h)) # norm B C dim
            # h = h.permute(2, 3, 0, 1) # [B C H W]
            # total_list[m][0] += torch.sum(h, dim=(0, 1), keepdim=True) # 在B维度上做element-wise的均值 [1 1 H W]

    for m in range(length):
        total_list[m][0] = copy.deepcopy(total_list[m][0] / total) # [1 C H W]
        # total_cls_list[m][0] = copy.deepcopy(total_cls_list[m][0] / total_mid_cls_cms[m]) # [1 c C H W]

    total = 0
    total_mid_cls_cms = [0] * length
    for i, data in enumerate(train_loader, 0):
        if i % 100 == 0: print_current_time(i)
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        logits, rec = model(x)
        total += y.size(0)

        for m in range(length):
            h = rec[m] # [B C H W]
            # feature map
            h = h.permute(0, 2, 3, 1) # [B H W C]
            h = h_std(args, h) # 应该是在C维度上做下LN（affine=False）+tanh
            h = h.permute(0, 3, 1, 2) # [B C H W]

            # ####################################################
            # y_onehot = F.one_hot(y, num_classes=logits.shape[-1]) # [B] -> [B c]
            # y_onehot = y_onehot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(args.device) # [B c] -> [B c 1 1 1]
            # h_ext = torch.unsqueeze(h, dim=1) # [B C H W] -> [B 1 C H W] 
            # total_mid_cls_cms[m] += torch.sum(y_onehot, dim=0, keepdim=True) # [1 c 1 1 1] 对应类别的个数
            # ####################################################


            for j in range(args.save_max_moment):
                if j > 0:
                    total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=0, keepdim=True)
                    # mid_cls_cms_add = torch.sum(torch.pow(y_onehot * h_ext - y_onehot * total_cls_list[m][0], j+1), dim=0, keepdim=True) # [1 c C H W]
                    # total_cls_list[m][j] += mid_cls_cms_add # [5][1 c C H W]
            # pixel
            # h = h.permute(2, 3, 0, 1) # [H W B C]
            # h = nn.Tanh()(nn.LayerNorm(h.shape[-2:], eps=1e-05, elementwise_affine=False)(h)) # norm B C dim
            # h = h.permute(2, 3, 0, 1) # [B C H W]
            # for j in range(args.save_max_moment):
            #     if j > 0:
            #         total_list[m][j] += torch.sum(torch.pow(h - total_list[m][0], j+1), dim=(0, 1), keepdim=True)

    for m in range(length):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total
                # total_cls_list[m][j] = total_cls_list[m][j] / total_mid_cls_cms[m]
    
    torch.save({'cmd_base_mid': total_list} , args.cm_file)