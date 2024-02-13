#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model vae --mode disentangle --kld_weight 1 --epochs 20 --dataset celeba --deterministic
