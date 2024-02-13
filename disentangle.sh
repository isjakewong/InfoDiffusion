#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python run.py --model diff --mode disentangle --a_dim 32 --mmd_weight 0.1 --epochs 20 --dataset celeba --deterministic
