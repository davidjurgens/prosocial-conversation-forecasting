#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=5

python3.7 main.py \
 --model_type albert \
 --training_type post \
 --model_name_or_path albert-base-v2 \
 --cache_dir /shared/0/projects/prosocial/data/yimingzhang/albert/cache \
 --train_files  /shared-1/1/projects/prosocial/finalized/train.text.tsv \
 --output_dir /shared/0/projects/prosocial/data/yimingzhang/albert/model \
 --per_gpu_train_batch_size 25 \
 --save_steps 3000

