#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python run.py --model diff --mode eval_fid --split_step 500 --a_dim 32 --mmd_weight 0.001 --batch_size 32 --sampling_number 10000 --epochs 30 --dataset celeba
