

# [B N D] -> [1 1 D] with cls token or mean pooling
def calc_cmd_layer_statistics_11D_clsmean(args, model, train_loader):
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

                # mu
                total_list[m][0] += torch.sum(h, dim=0, keepdim=True) # [1 1 D]

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
                
                # gaussian covariance matrix
                normlized_h = (h - total_list[m][0]).squeeze(1) # [B D]
                # print(normlized_h.shape)
                total_list[m][1] += (normlized_h.T @ normlized_h) # [D D]
                # print((normlized_h.T @ normlized_h).shape)

    for m in range(25):
        for j in range(args.save_max_moment):
            if j > 0:
                total_list[m][j] = total_list[m][j] / total
    
    torch.save({'cmd_base_mid': total_list} , args.cm_file)