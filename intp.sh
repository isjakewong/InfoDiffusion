#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python run.py --model diff --mode interpolate --a_dim 32 --mmd_weight 0.1 --epochs 20 --dataset celeba --deterministic
