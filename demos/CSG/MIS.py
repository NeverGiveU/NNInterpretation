import os
import torch
import resnet
import resnet_G
from collections import OrderedDict
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import os
import numpy as np 
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn import feature_selection


val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])),
    batch_size=128, 
    shuffle=False,
    num_workers=1, 
    pin_memory=True
)


#### 1. resnet20 with CSG-STD iteratively enabled
# ## network
# net = resnet_G.resnet20()
# cpt = torch.load("save_resnet20_G/model.th")

# n_cpt = OrderedDict()
# for k in cpt["state_dict"]:
#     n_cpt[k[7:]] = cpt["state_dict"][k]
# net.load_state_dict(n_cpt)
# net.eval()
# net.cuda()
# ##
# def encode_onehot(i, C=10):
#     t = torch.zeros((1, C)).float()
#     t[0, i] = 1.0
#     return t 
# ##
# cache = {
#     "a":[], # (N, K=64)
#     "p":[], # (N, C=10)
#     "1":[]  # (N, C=10)
# }
# with torch.no_grad():
#     for i, (input, target) in enumerate(val_loader):
#         target = target.cuda()
#         input_var = input.cuda()
#         target_var = target.cuda()
#         # compute output
#         output, activation = net(input_var)
#         cache["a"] += [activation.data.cpu()]
#         cache["p"] += [output.data.cpu().float()]
#         cache["1"] += [encode_onehot(t, C=10) for t in target.data.cpu()]
#     cache["a"] = torch.cat(cache["a"], 0)
#     cache["p"] = torch.cat(cache["p"], 0)
#     cache["1"] = torch.cat(cache["1"], 0)
#     for k in cache:
#         print(cache[k].size())
# ## MIS
# M = np.zeros((64, 10), dtype=np.float32)
# """
# 1) sklearn.feature selection.mutual_info_classif
# 2) MIS = mean_k max_c M_kc
# """
# for k in range(64):
#     for c in range(10):
#         a_vec = np.array(cache["a"][:, k]).reshape(-1, 1)
#         p_vec = np.array(cache["p"][:, c])
#         l_vec = np.array(cache["1"][:, c])
#         # use `sklearn` api
#         M[k, c] = feature_selection.mutual_info_classif(
#             a_vec, l_vec
#             )
#         # print(M[k, c])
# MIS = np.max(M, axis=1).mean()
# print("MIS:", MIS)


#### 1. resnet20 with only-STD enabled
## network
net = resnet.resnet20()
cpt = torch.load("save_resnet20/model.th")

n_cpt = OrderedDict()
for k in cpt["state_dict"]:
    n_cpt[k[7:]] = cpt["state_dict"][k]
net.load_state_dict(n_cpt)
net.eval()
net.cuda()
##
def encode_onehot(i, C=10):
    t = torch.zeros((1, C)).float()
    t[0, i] = 1.0
    return t 
##
cache = {
    "a":[], # (N, K=64)
    "p":[], # (N, C=10)
    "1":[]  # (N, C=10)
}
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        # compute output
        output, activation = net(input_var)
        cache["a"] += [activation.data.cpu()]
        cache["p"] += [output.data.cpu().float()]
        cache["1"] += [encode_onehot(t, C=10) for t in target.data.cpu()]
    cache["a"] = torch.cat(cache["a"], 0)
    cache["p"] = torch.cat(cache["p"], 0)
    cache["1"] = torch.cat(cache["1"], 0)
    for k in cache:
        print(cache[k].size())
## MIS
M = np.zeros((64, 10), dtype=np.float32)
"""
1) sklearn.feature selection.mutual_info_classif
2) MIS = mean_k max_c M_kc
"""
for k in range(64):
    for c in range(10):
        a_vec = np.array(cache["a"][:, k]).reshape(-1, 1)
        p_vec = np.array(cache["p"][:, c])
        l_vec = np.array(cache["1"][:, c])
        # use `sklearn` api
        M[k, c] = feature_selection.mutual_info_classif(
            a_vec, l_vec
            )
        # print(M[k, c])
MIS = np.max(M, axis=1).mean()
print("MIS:", MIS)