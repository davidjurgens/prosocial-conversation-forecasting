#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

export PRETRAINED_MODELS_ROOT_DIR=/shared/0/projects/prosocial/data/finalized/
export DATASET_ROOT_DIR=/shared/0/projects/prosocial/data/sample/

/opt/anaconda/bin/python examples/run_albert_eigenmetric.py \
--top_comment_pretrained_model_name_or_path albert-base-v2 \
--post_pretrained_model_name_or_path albert-base-v2 \
--classifier_dropout_prob 0.5 \
--subreddit_pretrained_path ${PRETRAINED_MODELS_ROOT_DIR}/subreddits/pretrained_subreddit_embeddings.tar.pth \
--input_dir ${DATASET_ROOT_DIR} \
--output_dir ${DATASET_ROOT_DIR}/checkpoints/run1 \
--learning_rate 1e-5 \
--n_epoch 5 \
--per_gpu_batch_size 4 \
--weight_decay 1e-6

# --num_subreddit_embeddings
# --subreddit_embeddings_size
# ${PRETRAINED_MODELS_ROOT_DIR}/finetuned_alberts/comment
# ${PRETRAINED_MODELS_ROOT_DIR}/finetuned_alberts/post
