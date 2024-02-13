#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python run.py --model vanilla --mode eval --a_dim 32 --epochs 10 --dataset celeba
