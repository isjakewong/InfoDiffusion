#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model diff --mode dis_quant --disent_metric tad --disent_num 10000 --mmd_weight 0.1 --a_dim 32 --epochs 10 --dataset fmnist --deterministic
