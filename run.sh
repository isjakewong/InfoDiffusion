#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model diff --mode train --mmd_weight 0.1 --a_dim 32 --epochs 50 --dataset celeba --batch_size 32 --save_epochs 5 --deterministic --prior regular --r_seed 64 
