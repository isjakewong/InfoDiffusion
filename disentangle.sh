#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model diff --mode disentangle --img_id 0 --mmd_weight 0.1 --a_dim 32 --epochs 20 --dataset celeba --deterministic --prior regular --r_seed 64
