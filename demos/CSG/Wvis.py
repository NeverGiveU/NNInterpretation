import os
import torch
import resnet
import resnet_G
from collections import OrderedDict
import numpy as np 
import matplotlib.pyplot as plt 


net = resnet_G.resnet20()
cpt = torch.load("save_resnet20_G/model.th") ## model trained under CSG-STD 

n_cpt = OrderedDict()
for k in cpt["state_dict"]:
    n_cpt[k[7:]] = cpt["state_dict"][k]
net.load_state_dict(n_cpt)

# print(net.G.size())
# print(net.G)

## plotting
fig, ax = plt.subplots(4, 1)
ax = ax.flatten()

G_arr = net.G.data.cpu()
print("L1 density:", np.abs(G_arr).sum()/64/10) # L1 density
G_arr = np.array(G_arr)
G_arr = G_arr - G_arr.min()
G_arr = G_arr / G_arr.max()
# plt.subplot(311)
ax0 = ax[0].imshow(G_arr, cmap=plt.cm.jet)

W_arr = net.linear.weight.data.cpu()
W_arr = np.array(W_arr)
W_arr = W_arr - W_arr.min()
W_arr = W_arr / W_arr.max()
# plt.subplot(312)
ax1 = ax[1].imshow(W_arr, cmap=plt.cm.jet)

C_arr = G_arr * W_arr 
C_arr = C_arr / C_arr.max()
# plt.subplot(313)
ax3 = ax[3].imshow(C_arr, cmap=plt.cm.jet)

## 
net = resnet.resnet20()
cpt = torch.load("save_resnet20/model.th") ## model trained under STD-only

n_cpt = OrderedDict()
for k in cpt["state_dict"]:
    n_cpt[k[7:]] = cpt["state_dict"][k]
net.load_state_dict(n_cpt)

W_arr = net.linear.weight.data.cpu()
W_arr = np.array(W_arr)
W_arr = W_arr - W_arr.min()
W_arr = W_arr / W_arr.max()
# plt.subplot(312)
ax2 = ax[2].imshow(W_arr, cmap=plt.cm.jet)

plt.colorbar(ax0, ax=ax[0], fraction=0.01)
plt.colorbar(ax1, ax=ax[1], fraction=0.01)
plt.colorbar(ax2, ax=ax[2], fraction=0.01)
plt.colorbar(ax3, ax=ax[3], fraction=0.01)

plt.show()
