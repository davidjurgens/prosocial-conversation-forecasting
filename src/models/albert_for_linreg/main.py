import os
import argparse
import logging
import shutil

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from transformers.optimization import AdamW
from tensorboardX import SummaryWriter


from src.models.albert_for_linreg.utils import load_and_cache_dataset, load_coeffs
from src.models.albert_for_linreg.model import LinRegModel
from src.models.albert_for_linreg.train import train
from src.models.albert_for_linreg.eval import evaluate

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_architecture', choices=['comment', 'comment_post'], required=True
    )
    parser.add_argument(
        '--base_model', default='albert-base-v2'
    )
    parser.add_argument(
        '--comment_model', default='albert-base-v2', type=str)
    parser.add_argument(
        '--post_model', default='albert-base-v2', type=str)
    parser.add_argument(
        "--batch_size", default=20, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument(
        "--sequence_length", default=128, type=int, help="Sequence length for language model."
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Total number of training epochs to perform."
    )
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training_steps", type=int, default=0,
     help="Evaluate on dev set every X steps during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
     help="Backprop loss every X steps.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        required=True,
        type=str
    )
    parser.add_argument(
        "--output_dir",
        default="",
        required=True,
        type=str
    )
    parser.add_argument(
        "--output_postfix",
        default=None,
        type=str
    )
    parser.add_argument(
        "--linear_regression_coefs",
        default=None,
        type=str
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
        
    args.output_dir = os.path.join(args.output_dir, args.model_architecture)
    if args.output_postfix:
        args.output_dir = os.path.join(args.output_dir, args.output_postfix)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    tb_path = os.path.join(args.output_dir, 'tb')
    tb_writer = SummaryWriter(tb_path)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, 
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=True)

    do_post = bool(args.model_architecture=='comment_post')

    train_dataset = load_and_cache_dataset(args, tokenizer, 'train', post=do_post)
    dev_dataset = load_and_cache_dataset(args, tokenizer, 'dev', post=do_post)
    test_dataset = load_and_cache_dataset(args, tokenizer, 'test', post=do_post)


    model = LinRegModel(load_coeffs(args), args).to(args.device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    if args.do_train:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        model = train(args, model, train_dataset, dev_dataset, optimizer, tb_writer)



    if args.do_eval:
        evaluate(args, model, test_dataset, tb_writer)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save, os.path.join(args.output_dir, 'model.pt'))
    tb_writer.close()



if __name__ == "__main__":
    main()

