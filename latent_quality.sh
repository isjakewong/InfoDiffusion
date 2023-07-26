#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model diff --mode latent_quality --a_dim 256 --mmd_weight 0.1 --epochs 50 --dataset celeba --sampling_number 16 --deterministic --prior regular --r_seed 64
