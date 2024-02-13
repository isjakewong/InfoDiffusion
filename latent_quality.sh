#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python run.py --model diff --mode latent_quality --a_dim 32 --mmd_weight 0.1 --epochs 20 --dataset celeba --deterministic --sampling_number 16
