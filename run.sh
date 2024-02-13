#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python run.py --model diff --mode train --a_dim 32 --epochs 50 --dataset celeba --batch_size 20 --display_epochs 10
