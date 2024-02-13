#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model vanilla --mode train --a_dim 32 --epochs 50 --dataset celeba --batch_size 128 --display_epochs 10
