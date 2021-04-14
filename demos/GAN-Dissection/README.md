# GAN Dissection Step-by-Step

### 🙈 Profile

> In this project, we re-implement **dissections** on GANs. 
>
> As the official codes are complicated and not easy for reading and understanding.
>
> **Referred papers** includes:
>
> 1️⃣ Understanding the Role of Individual Units in a Deep Neural Networks. [[Paper📜](https://www.pnas.org/content/117/48/30071), [Blog📝](https://blog.csdn.net/WinerChopin/article/details/108672765?spm=1001.2014.3001.5501) [Web💻](http://dissect.csail.mit.edu/)]
>
> 2️⃣ GAN Dissection: Visualizing and Understanding Generative Adversarial Networks. [[📜](https://arxiv.org/abs/1811.10597)]
>
> 3️⃣ Semantic Photo Manipulation with a Generative Image Prior. [[📜](https://arxiv.org/abs/2005.07727v1)]

### 🐞Introduction

> Thanks [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) for the `tf2pytorch` convertor. 
>
> Thanks [stylegan2-tensorflow](https://github.com/NVlabs/stylegan2) for the pre-trained model on LSUN-church. 🐂🍺

> We share both `tf` and `pytorch` models, and you can download from [👇]() and save them in `pretrained_models`.

### 🐞Step-by-Step

> **Step 1**: Import packages and Initialize the model.
>
> ```python
> import torch
> import os
> from model import Generator
> import numpy as np 
> from tqdm import tqdm
> from PIL import Image
> import matplotlib.pyplot as plt 
> from torchvision import transforms
> 
> 
> layer_id = 2 # explore in the `2nd` layer
> ## prepare models
> g = Generator(256, 512, 8, 2, layer_id=layer_id)
> ema_ckpt = torch.load("pretrained_models/stylegan2-church-config-f.pt")
> g.load_state_dict(ema_ckpt["g_ema"])
> g.cuda()
> latent_avg = ema_ckpt["latent_avg"].data.cuda()
> ```
>
> **Step 2**: Synthesis some samples and save both the input `z` and output `image`.
>
> ```python
> ## randomly generate some examples
> ## you may not need to do that as we have provide some examples in our exp
> if os.path.exists("demo/church") is not True:
>  os.mkdir("demo/church")
>  os.mkdir("demo/church/z")
>  os.mkdir("demo/church/img")
>  os.mkdir("demo/church/mask")
> 
> n_samples = 16
> z = np.random.RandomState(0).randn(n_samples, 512).astype("float32")
> z = torch.from_numpy(z).cuda()
> 
> with torch.no_grad():
>  opt, _ = g(
>      [z],
>      truncation=0.5,
>      truncation_latent=latent_avg,
>      randomize_noise=False,
>  )
> 
> ## save
> for i in tqdm(range(n_samples)):
>  z0 = z[i].data.cpu()
>  opt0 = opt[i].data.cpu().permute(1, 2, 0)
>  opt0 = opt0*0.5+0.5
>  opt0 = np.array(opt0*255).astype(np.uint8)
>  img0 = Image.fromarray(opt0).convert("RGB")
> 
>  torch.save(z0, "demo/church/z/{}.pt".format(i))
>  img0.save("demo/church/img/{}.png".format(i))
> print("Finish!")
> ```
>
> **Step 3**: Select an interested image and draw a binary mask for interested regions with PS or SAI.
>
> For example, we select `0.png` and we have:
>
> ![image](https://github.com/NeverGiveU/NNInterpretation/main/demos/GAN-Dissection/demo/church/img/8.png)![mask](https://github.com/NeverGiveU/NNInterpretation/main/demos/GAN-Dissection/demo/church/mask/8.png).
>
> Load in the images as
>
> ```python
> ## load in images
> fname = "8"
> z = torch.load("demo/church/z/{}.pt".format(fname)).unsqueeze(0).cuda()
> img = Image.open("demo/church/img/{}.png".format(fname)).convert("RGB")
> plt.subplot(231)
> plt.imshow(img)
> mask = Image.open("demo/church/mask/{}.png".format(fname)).convert("L")
> 
> with torch.no_grad():
>     opt, feat = g( 
>         [z],
>         truncation=0.5,
>         truncation_latent=latent_avg,
>         randomize_noise=False,
>     )
> opt = opt[0].data.cpu().permute(1,2,0)
> plt.subplot(232), plt.imshow((np.array(opt*0.5+0.5)*255).astype(np.uint8))
> ```
>
> Note that, we have modified the `model.py` and the features of the `layer_id`<sup>th</sup> are returned as `feat`.
>
> **Step 4**: Check  & Compare the 1️⃣re-synthesized image `opt` and 2️⃣the original image `img` and 3️⃣re-constructed image `fi_recon` from `feat`. Make sure that they are the same.
>
> ```python
> fi_recon = g.final_forward(feat, layer_id=layer_id)
> 
> fi_recon = fi_recon[0].data.cpu().permute(1,2,0)
> plt.subplot(233), plt.imshow((np.array(fi_recon*0.5+0.5)*255).astype(np.uint8))
> ```
>
> **Step 5**: Compute the correlation between the selected region (or object or semantic), and each channel.
>
> Specially, 
>
> 1️⃣ For image `x`, resize the `mask` to be in the same size of `feat`.
>
> ```python
> feat = feat[0].cpu() # (C, H, W)
> C, h, w = feat.size()
> 
> t = transforms.Resize((h, w))
> mask = t(mask)
> mask = torch.from_numpy(np.array(mask))
> ```
>
> 2️⃣ for each channel `u`, for each pixel `p`, the activated value is `a`<sub>`u`</sub>`(x, p)`; 
>
> we want to find such a threshold `t`<sub>`u`</sub>, such that
> $$
> t_u:=\max_t \mathbb P_{x,p} [a_u(x,p)>t]>0.01
> $$
> In other words, we want to sort all the activated values in ascending order, then get the top 1% of the pixels.
>
> (我们希望将所有像素按照激活值从大到小排序后，取前 1 % 的像素点。)
>
> Then we can compute this "correlation" in an `IoU`-form as
> $$
> {\rm IoU}_{u,c}:={{\mathbb P_{x,p}[s_c(x,p)∧(a_u(x,p)>t_u)]}\over{\mathbb P_{x,p}[s_c(x,p)∨(a_u(x,p)>t_u)]}}
> $$
>
> > 👆上面的公式表示：某一类别 `c` 的对象占有图像像素的区域与单元 `u` 激活前 1 % 激活值占领的区域，两者之间的 `IoU` 值。
> > 注意，实际实验中，`c` 不仅指具体的类别，还可以是不同的颜色（`color`），部分（`part`），材料（`material`）。
>
> Specially, we can implement these steps as
>
> ```python
> field1 = 0.01
> id1 = h*w-int(field1*h*w)
> 
> ious = []
> for c in range(C):
>     # feat_ch = feat[c]
>     feat_ch = torch.abs(feat[c]) # you can choose whether to use the absolute value.
>     
>     feat_vec = torch.sort(feat_ch.view(-1))[0]
>     feat_ch_mask = (feat_ch >= feat_vec[id1])
>     ious += [(mask*feat_ch_mask).sum().item() / (h*w)]
> 
> sorted_idcies = sorted(range(C), key=lambda i: ious[i]) # sorted channel indices
> ious.sort() # sorted correlations
> ```
>
> **Step 6**: Respectively, set the highest 4, 16, & 128channels to **ZERO**, and re-forward the propagation.
>
> ```python
> i = 1
> feat = feat.cuda().unsqueeze(0)
> 
> for Clen in (4, 16, 128):
>     vec_mask = torch.ones(C)
>     for c in range(Clen):
>         vec_mask[sorted_idcies[C-1-c]] = 0.0
>     vec_mask = vec_mask.cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
>     masked_feat = vec_mask * feat 
>     masked_fi_recon = g.final_forward(masked_feat, layer_id)
> 
>     # plot
>     masked_fi_recon = masked_fi_recon[0].data.cpu().permute(1,2,0)
>     plt.subplot(2,3,i+3)
>     plt.imshow((np.array(masked_fi_recon*0.5+0.5)*255).astype(np.uint8))
>     i += 1
> ```
>
> **Step 7**: Plot with `plt.show()`.
>
> ![result](https://github.com/NeverGiveU/NNInterpretation/main/demos/GAN-Dissection/sample_dissected_results.png)
>
> As discussed in the paper, as the important channels of the two headmost trees are "turned off", the obscured shorter trees are shown, while the other objects and semantics keep unchanged.

