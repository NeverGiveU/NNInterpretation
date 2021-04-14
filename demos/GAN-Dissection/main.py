import torch
import os
from model import Generator
import numpy as np 
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms


layer_id = 2 # explore in the `2nd` layer
## prepare models
g = Generator(256, 512, 8, 2, layer_id=layer_id)
ema_ckpt = torch.load("pretrained_models/stylegan2-church-config-f.pt")
g.load_state_dict(ema_ckpt["g_ema"])
g.cuda()
latent_avg = ema_ckpt["latent_avg"].data.cuda()


# ## randomly generate some examples
# if os.path.exists("demo/church") is not True:
#     os.mkdir("demo/church")
#     os.mkdir("demo/church/z")
#     os.mkdir("demo/church/img")
#     os.mkdir("demo/church/mask")

# n_samples = 16
# z = np.random.RandomState(0).randn(n_samples, 512).astype("float32")
# z = torch.from_numpy(z).cuda()

# with torch.no_grad():
#     opt, _ = g(
#         [z],
#         truncation=0.5,
#         truncation_latent=latent_avg,
#         randomize_noise=False,
#     )

# ## save
# for i in tqdm(range(n_samples)):
#     z0 = z[i].data.cpu()
#     opt0 = opt[i].data.cpu().permute(1, 2, 0)
#     opt0 = opt0*0.5+0.5
#     opt0 = np.array(opt0*255).astype(np.uint8)
#     img0 = Image.fromarray(opt0).convert("RGB")

#     torch.save(z0, "demo/church/z/{}.pt".format(i))
#     img0.save("demo/church/img/{}.png".format(i))
# print("Finish!")


## 
fname = "8"
z = torch.load("demo/church/z/{}.pt".format(fname)).unsqueeze(0).cuda()
img = Image.open("demo/church/img/{}.png".format(fname)).convert("RGB")
plt.subplot(231)
plt.imshow(img)
mask = Image.open("demo/church/mask/{}.png".format(fname)).convert("L")

with torch.no_grad():
    opt, feat = g(
        [z],
        truncation=0.5,
        truncation_latent=latent_avg,
        randomize_noise=False,
    )

## compute IoU
fi_recon = g.final_forward(feat, layer_id=layer_id)

feat = feat[0].cpu() # (C, H, W)
C, h, w = feat.size()

t = transforms.Resize((h, w))
mask = t(mask)
mask = torch.from_numpy(np.array(mask))

field1 = 0.01
id1 = h*w-int(field1*h*w)

ious = []
for c in range(C):
    # feat_ch = feat[c]
    feat_ch = torch.abs(feat[c])
    # print(mask.size(), feat_ch.size())
    # 1. sort
    feat_vec = torch.sort(feat_ch.view(-1))[0]
    feat_ch_mask = (feat_ch >= feat_vec[id1])
    ious += [(mask*feat_ch_mask).sum().item() / (h*w)]

    # break
# print(ious)
sorted_idcies = sorted(range(C), key=lambda i: ious[i])
ious.sort()

# 
i = 1
feat = feat.cuda().unsqueeze(0)

# for Clen in (256, 128, 96):
for Clen in (16, 128, 224):
    # vec_mask = torch.zeros(C)
    vec_mask = torch.ones(C)
    for c in range(Clen):
        vec_mask[sorted_idcies[C-1-c]] = 0.0#1.0
    vec_mask = vec_mask.cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    masked_feat = vec_mask * feat 
    masked_fi_recon = g.final_forward(masked_feat, layer_id)

    # plot
    masked_fi_recon = masked_fi_recon[0].data.cpu().permute(1,2,0)
    plt.subplot(2,3,i+3)
    plt.imshow((np.array(masked_fi_recon*0.5+0.5)*255).astype(np.uint8))
    i += 1


opt = opt[0].data.cpu().permute(1,2,0)
plt.subplot(232)
plt.imshow((np.array(opt*0.5+0.5)*255).astype(np.uint8))

fi_recon = fi_recon[0].data.cpu().permute(1,2,0)
plt.subplot(233)
plt.imshow((np.array(fi_recon*0.5+0.5)*255).astype(np.uint8))


plt.show()