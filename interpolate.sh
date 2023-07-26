#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model diff --mode interpolate --mmd_weight 0.1 --img_id 0 --a_dim 32 --epochs 50 --dataset celeba --deterministic --prior regular --r_seed 64
