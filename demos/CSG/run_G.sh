#!/bin/bash

for model in resnet20 # resnet32 resnet44 resnet56 resnet110 resnet1202
do
    echo "CUDA_VISIBLE_DEVICES=2 python -u trainer_G.py  --arch=$model  --save-dir=save_$model_G"
    CUDA_VISIBLE_DEVICES=2 python -u trainer_G.py  --arch=$model  --save-dir=save_$model_G
done