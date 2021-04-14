# Training Interpretable Convolutional Neural Networks by Dierentiating Class-specic Filters
### 🙈 Profile

> 📜 [Paper](http://arxiv.org/abs/2007.08194v1) or [Paper w/i my notes](assets/Training Interpretable Convolutional NeuralNetwor.pdf)
>
> 📺[PPt(My own Presentation)](assets/Training Interpretable Convolutional Neural.pptx)
>
> ---
>
> Yep, this is a work for a course project. I just found neither official nor public codes for this paper... 

### 🐞Introduction

> The project is based on [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Very nice encapsulation! 🐂🍺

#### Part 1

> Till **2021/2/23**, we just did a very simple exploration on `resnet20` for image classification on `cifar10`. 
>
> The results is shown as the following tables. For each item, "**val_in_paper**(**val_our_implementation**)". And we compare our implementation to the results proposed in paper.

**Table 1.** Metrics of the STD CNN (baseline) and the CSG CNN (Ours).

| Dataset  |  Model   | *C*  | *K*  | Training |      Accuracy      |        MIS         |     L1-density     | L1-interval  |
| :------: | :------: | :--: | :--: | :------: | :----------------: | :----------------: | :----------------: | :----------: |
| CIFAR-10 | ResNet20 |  10  |  64  |   CSG    | **0.9192**(0.9177) | **0.1603**(0.1264) | **0.0788**(0.1000) | [0.01, 0.01] |
| CIFAR-10 | ResNet20 |  10  |  64  |   STD    | 0.9169(**0.9194**) | 0.1119(**0.0883**) |         -          |      -       |

> We also visualize the matrices as **Fig.4** in paper.

![001](https://github.com/NeverGiveU/NNInterpretation/main/demos/CSG/assets/Gcsg-Wcsg-Wstd-GcsgxWcsg.png)

> The implementation is included in **THREE** `*_G.py` files.
>
> `MIS.py` is used to calculate the `MIS` score.
>
> `Wvis.py` is used to plot the over figure.
>
> RUN the experiment using `sh run_G.sh` or `sh run.sh`

#### Part 2

> May come in future ..😀