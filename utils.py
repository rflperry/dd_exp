import math
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch

import copy
import random
import concurrent.futures

## Distributions 

def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
#     means = [[-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [-1.5, 1.5]]
    means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=int(n / 4)
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate([np.ones((int(n / 4))) * int(i < 2) for i in range(len(means))])
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )
    return X, Y.astype(int)
        

## Network functions

# Model 
class Net(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False, penultimate=False, bn=False):
        super(Net, self).__init__()

        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, hidden_size, bias=bias))

        for ll in range(n_hidden):
            module.append( activation )
            if bn:
                module.append( nn.BatchNorm1d( hidden_size ) )
            module.append( nn.Linear(hidden_size, hidden_size, bias=bias) )      
        
        if penultimate:
            module.append( activation )
            if bn:
                module.append( nn.BatchNorm1d( hidden_size ) )
            module.append( nn.Linear(hidden_size, 2, bias=bias) )
            hidden_size = 2
            
        module.append( activation )
        if bn:
            module.append( nn.BatchNorm1d( hidden_size ) )
        module.append( nn.Linear(hidden_size, out_dim, bias=bias) )

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)

# functions
def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

def train_model(model, train_x, train_y, multi_label=False, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    losses = []
        
    for step in range(1000):
        optimizer.zero_grad()
        outputs = model(train_x)
        if multi_label:
            train_y = train_y.type_as(outputs)
        
        loss=loss_func(outputs, train_y)
        trainL = loss.detach().item()
        if verbose and (step % 500 == 0):
            print("train loss = ", trainL)
        losses.append(trainL)
        loss.backward()
        optimizer.step()
    
    return losses
                
def get_model(hidden_size=20, n_hidden=5, in_dim=2, out_dim=1, penultimate=False, use_cuda=True, bn=False):
    in_dim = in_dim
    out_dim = out_dim #1
    model = Net(in_dim, out_dim, n_hidden=n_hidden, hidden_size=hidden_size,
                activation=torch.nn.ReLU(), bias=True, penultimate=penultimate, bn=bn)
    
    if use_cuda:
        model=model.cuda()
        
    return model

            
def get_dataset(N=1000, one_hot=False, cov_scale=1, include_hybrid=False):
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
    if include_hybrid:
        D_x, D_y = generate_gaussian_parity(cov_scale=cov_scale, n=2*N, angle_params=0)
        D_perm = np.random.permutation(2*N)
        D_x, D_y  = D_x[D_perm,:], D_y[D_perm]
        train_x, train_y = D_x[:N], D_y[:N]
        ghost_x, ghost_y = D_x[N:], D_y[N:]
        hybrid_sets = []
        rand_idx = random.sample(range(0,N-1), N//10)
        for rand_i in rand_idx:
            hybrid_x, hybrid_y = np.copy(train_x), np.copy(train_y)
            hybrid_x[rand_i], hybrid_y[rand_i] = ghost_x[rand_i], ghost_y[rand_i]
            hybrid_x = torch.FloatTensor(hybrid_x)
            hybrid_y = (torch.FloatTensor(hybrid_y).unsqueeze(-1))
            hybrid_x, hybrid_y = hybrid_x.cuda(), hybrid_y.cuda()
            hybrid_sets.append((hybrid_x, hybrid_y))
    else:
        train_x, train_y = generate_gaussian_parity(cov_scale=cov_scale, n=N, angle_params=0)
        train_perm = np.random.permutation(N)
        train_x, train_y = train_x[train_perm,:], train_y[train_perm] 
    test_x, test_y = generate_gaussian_parity(cov_scale=cov_scale, n=2*N, angle_params=0)
    
    test_perm = np.random.permutation(2*N)
    test_x, test_y  = test_x[test_perm,:], test_y[test_perm]
    
    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)

    train_y = (torch.FloatTensor(train_y).unsqueeze(-1))#[:,0]
    test_y = (torch.FloatTensor(test_y).unsqueeze(-1))#[:,0]
    
    if one_hot:
        train_y = torch.nn.functional.one_hot(train_y[:,0].to(torch.long))
        test_y = torch.nn.functional.one_hot(test_y[:,0].to(torch.long))
    
    # move to gpu
    if use_cuda:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()
        
    if include_hybrid:
        return train_x, train_y, test_x, test_y, hybrid_sets
    
    return train_x, train_y, test_x, test_y

                          
def run_experiment(depth, iterations, reps=100, width=3, cov_scale=1):
    result = lambda: None
    
    xx, yy = np.meshgrid(np.arange(-2, 2, 4 / 100), np.arange(-2, 2, 4 / 100))
    true_posterior = np.array([pdf(x) for x in (np.c_[xx.ravel(), yy.ravel()])])
    
    rep_full_list = []
    imgs = []
#     train_x, train_y, test_x, test_y = get_dataset(cov_scale=cov_scale)
    train_x, train_y, test_x, test_y, hybrid_sets = get_dataset(N=1000, cov_scale=cov_scale, include_hybrid=True)
    depth = depth
    penultimate_vars_reps = []
    for rep in range(reps):#25
        print('rep: ' + str(rep))
        # Shffle train set labels
        train_y_tmp = torch.clone(train_y)
        train_y[train_y_tmp==0] = 1
        train_y[train_y_tmp==1] = 0
        test_y_tmp = torch.clone(test_y)
        test_y[test_y_tmp==0] = 1
        test_y[test_y_tmp==1] = 0

        del train_y_tmp
        losses_list = []
        num_pars = []
        
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        for i in range(1, iterations):#20
            print('now running', i)

            ## Increasing Depth
            if depth:
                if i < 5:
                    model = get_model(n_hidden = i, hidden_size=i, penultimate=False, bn=False)
                else:
                    model = get_model(n_hidden = i, penultimate=False, bn=False)
            else:
            ## Increasing Width
                model = get_model(hidden_size = i, n_hidden=width, penultimate=False, bn=False)

            n_par = sum(p.numel() for p in model.parameters())

            losses = train_model(model, train_x, train_y)


            if depth:
                n_nodes = i*20 if i>5 else i*i
            else:
                n_nodes = i*3
            
            with torch.no_grad():
                pred_train, pred_test = model(train_x), model(test_x)
                
                train_y = train_y.type_as(pred_train)
                test_y  = test_y.type_as(pred_test)
                train_loss = torch.nn.BCEWithLogitsLoss()(pred_train, train_y)
    #             train_acc = (torch.argmax(pred_train,1) == torch.argmax(train_y,1)).sum().cpu().data.numpy().item() / train_y.size(0)
                train_acc = (torch.sigmoid(pred_train).round() == train_y).sum().cpu().data.numpy().item() / train_y.size(0)
                test_loss = torch.nn.BCEWithLogitsLoss()(pred_test, test_y)
    #             test_acc = (torch.argmax(pred_test,1) == torch.argmax(test_y,1)).sum().cpu().data.numpy().item() / test_y.size(0)
                test_acc = (torch.sigmoid(pred_test).round() == test_y).sum().cpu().data.numpy().item() / test_y.size(0)

            losses_list.append(losses)
            num_pars.append(n_par)

            train_loss_list.append(train_loss.item())
            test_loss_list.append(test_loss.item())
            train_acc_list.append(1-train_acc)
            test_acc_list.append(1-test_acc)                                                                                                                
            fname = ("depth" if depth else "width") + (str(width) if width != 3 else "") + "_cov" + str(cov_scale) + "_" + str(i)
        rep_full_list.append([losses_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list])
        
    result.num_pars = num_pars
    [result.full_loss_list, result.test_loss_list, result.train_loss_list, result.test_err_list, result.train_err_list] = extract_losses(rep_full_list)

    return result 

# Losses
def extract_losses(rep_full_list):
    fl_list = []
    for losses_list, *_ in rep_full_list:
        final_loss = [l[-1] for l in losses_list]
        fl_list.append(final_loss)

    full_loss_list = np.array(fl_list)
    test_loss_list = np.array([ee[2] for ee in rep_full_list])
    train_loss_list = np.array([ee[1] for ee in rep_full_list])
    test_err_list = np.array([ee[4] for ee in rep_full_list])
    train_err_list = np.array([ee[3] for ee in rep_full_list])

    return [full_loss_list, test_loss_list, train_loss_list, test_err_list, train_err_list]
