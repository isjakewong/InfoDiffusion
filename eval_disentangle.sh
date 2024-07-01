#!/bin/bash
python eval_disentanglement.py --model diff --a_dim 256 --mmd_weight 0.1 --epochs 50 --dataset celeba --sampling_number 16 --deterministic --prior regular --r_seed 64
