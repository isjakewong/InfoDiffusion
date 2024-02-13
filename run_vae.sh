#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python run.py --model vae --mode train --a_dim 32 --epochs 30 --dataset celeba --batch_size 64 --display_epochs 10
