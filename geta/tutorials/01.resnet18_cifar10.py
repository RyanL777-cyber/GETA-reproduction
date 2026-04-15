#!/usr/bin/env python
# coding: utf-8

# ## Tutorial 1. ResNet18 on CIFAR10. 
# 
# 
# In this tutorial, we will show 
# 
# - How to end-to-end train and compress a ResNet18 from scratch on CIFAR10 to get a compressed ResNet18.
# - The compressed ResNet18 achives both **high performance** and **significant FLOPs and parameters reductions** than the full model. 
# - The compressed ResNet18 **reduces about 92% parameters** to achieve **92.91% accuracy** only lower than the baseline by **0.11%**.
# - More detailed new HESSO optimizer setup. (Technical report regarding HESSO will be released on the early of 2024).

# ### Step 1. Create OTO instance

# In[1]:


import sys
sys.path.append('..')
from sanity_check.backends.resnet_cifar10 import resnet18_cifar10
from only_train_once import OTO
import torch

model = resnet18_cifar10()
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())


# #### (Optional) Visualize the pruning dependancy graph of DNN

# In[2]:


# A ResNet_zig.gv.pdf will be generated to display the depandancy graph.
oto.visualize(view=False, out_dir='../cache')


# ### Step 2. Dataset Preparation

# In[3]:


from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

trainset = CIFAR10(root='cifar10', train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
testset = CIFAR10(root='cifar10', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader =  torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)


# ### Step 3. Setup HESSO optimizer
# 
# The following main hyperparameters need to be taken care.
# 
# - `variant`: The optimizer that is used for training the baseline full model. Currently support `sgd`, `adam` and `adamw`.
# - `lr`: The initial learning rate.
# - `weight_decay`: Weight decay as standard DNN optimization.
# - `target_group_sparsity`: The target group sparsity, typically higher group sparsity refers to more FLOPs and model size reduction, meanwhile may regress model performance more.
# - `start_pruning_steps`: The number of steps that **starts** to prune.
# - `pruning_steps`: The number of steps that **finishes** pruning (reach `target_group_sparsity`) after `start_pruning_steps`.
# - `pruning_periods`:  Incrementally produce the group sparsity equally among pruning periods.
# 
# We empirically suggest `start_pruning_steps` as 1/10 of total number of training steps. `pruning_steps` until 1/4 or 1/5 of total number of training steps.
# The advatnages of HESSO compared to DHSPG is its explicit control over group sparsity exploration, which is typically more convenient.

# In[4]:


optimizer = oto.hesso(
    variant='sgd', 
    lr=0.1, 
    weight_decay=1e-4,
    target_group_sparsity=0.7,
    start_pruning_step=10 * len(trainloader), 
    pruning_periods=10,
    pruning_steps=10 * len(trainloader)
)


# ### Step 4. Train ResNet18 as normal.

# In[5]:


from utils.utils import check_accuracy

max_epoch = 100
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
# Every 50 epochs, decay lr by 10.0
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) 

for epoch in range(max_epoch):
    f_avg_val = 0.0
    model.train()
    lr_scheduler.step()
    for X, y in trainloader:
        X = X.cuda()
        y = y.cuda()
        y_pred = model.forward(X)
        f = criterion(y_pred, y)
        optimizer.zero_grad()
        f.backward()
        f_avg_val += f
        optimizer.step()
    opt_metrics = optimizer.compute_metrics()
    # group_sparsity, param_norm, _ = optimizer.compute_group_sparsity_param_norm()
    # norm_important, norm_redundant, num_grps_important, num_grps_redundant = optimizer.compute_norm_groups()
    accuracy1, accuracy5 = check_accuracy(model, testloader)
    f_avg_val = f_avg_val.cpu().item() / len(trainloader)

    print("Ep: {ep}, loss: {f:.2f}, norm_all:{param_norm:.2f}, grp_sparsity: {gs:.2f}, acc1: {acc1:.4f}, norm_import: {norm_import:.2f}, norm_redund: {norm_redund:.2f}, num_grp_import: {num_grps_import}, num_grp_redund: {num_grps_redund}"\
         .format(ep=epoch, f=f_avg_val, param_norm=opt_metrics.norm_params, gs=opt_metrics.group_sparsity, acc1=accuracy1,\
         norm_import=opt_metrics.norm_important_groups, norm_redund=opt_metrics.norm_redundant_groups, \
         num_grps_import=opt_metrics.num_important_groups, num_grps_redund=opt_metrics.num_redundant_groups
        ))


# ### Step 5. Get compressed model in torch format

# In[6]:


# By default OTO will construct subnet by the last checkpoint. If intermedia ckpt reaches the best performance,
# need to reinitialize OTO instance
# oto = OTO(torch.load(ckpt_path), dummy_input)
# then construct subnetwork
oto.construct_subnet(out_dir='./cache')


# ### (Optional) Check the compressed model size

# In[7]:


import os

full_model_size = os.stat(oto.full_group_sparse_model_path)
compressed_model_size = os.stat(oto.compressed_model_path)
print("Size of full model     : ", full_model_size.st_size / (1024 ** 3), "GBs")
print("Size of compress model : ", compressed_model_size.st_size / (1024 ** 3), "GBs")


# ### (Optional) Check the compressed model accuracy
# #### # Both full and compressed model should return the exact same accuracy.

# In[8]:


full_model = torch.load(oto.full_group_sparse_model_path)
compressed_model = torch.load(oto.compressed_model_path)

acc1_full, acc5_full = check_accuracy(full_model, testloader)
print("Full model: Acc 1: {acc1}, Acc 5: {acc5}".format(acc1=acc1_full, acc5=acc5_full))

acc1_compressed, acc5_compressed = check_accuracy(compressed_model, testloader)
print("Compressed model: Acc 1: {acc1}, Acc 5: {acc5}".format(acc1=acc1_compressed, acc5=acc5_compressed))

