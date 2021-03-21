"""Author: Yiming Zhang"""
import os
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          AlbertConfig, AlbertForMaskedLM, AlbertTokenizer)
import torch
from torch.utils.data import ConcatDataset, TensorDataset
import numpy as np
import random
import pickle
import logging
import re
import glob
import shutil
import pandas as pd


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'albert' : (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def load_and_cache_datasets(args, tokenizer):
    if isinstance(args.train_files, list):
        datasets = []
        for f in args.train_files:
            dataset = get_dataset(tokenizer, args, f)
            datasets.append(dataset)
        
        return ConcatDataset(datasets)
    else:
        raise ValueError


def get_dataset(tokenizer, args, file_path):
    assert(os.path.isfile(file_path))

    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(
        args.cache_dir, args.training_type + '_' + args.model_name_or_path + '_cached_lm_' + filename)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'rb') as handle:
            return pickle.load(handle)
        assert(0)
    else:
        
        logger.info("Creating features from dataset file at %s", file_path)
        dataset = load_text_from_tsv(args, file_path, tokenizer)
        logger.info("Saving features into cached file %s",
                    cached_features_file)
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(dataset, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return dataset

def load_text_from_tsv(args, f, tokenizer):
    column_name = 'Top_comment_text' if args.training_type == "comment" else "Post_text"
    df = pd.read_csv(f, usecols=[column_name] ,sep='\t')
    d = tokenizer.batch_encode_plus(df[column_name], max_length=args.sequence_length,
     pad_to_max_length=True, return_attention_masks=True, return_token_type_ids=False)
    
    return TensorDataset(torch.LongTensor(d['input_ids']), torch.FloatTensor(d['attention_mask']))

def get_mask_token_id(args, tokenizer):
    return tokenizer.mask_token_id


def mask_tokens(tokens, mask, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    mask = mask.bool()
    mask_token_id = get_mask_token_id(args, tokenizer)
    labels = tokens.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(
        val, already_has_special_tokens=True) for val in tokens.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix[special_tokens_mask | ~mask] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # We only compute loss on masked tokens
    # TODO: Figure out why -1 does not work with xlm
    labels[~masked_indices] = -100
    

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = mask_token_id 

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

    tokens[indices_random] = torch.randint(
        len(tokenizer), (indices_random.sum(),), dtype=tokens.dtype)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels


def mask_tokens_legacy(tokens, mask, tokenizer, args):
    def invert(t):
        return 1-t
    mask_token_id = get_mask_token_id(args, tokenizer)
    labels = tokens.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    mask = mask.to(torch.uint8)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(
        val, already_has_special_tokens=True) for val in tokens.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.uint8)
    
    probability_matrix[special_tokens_mask | invert(mask)] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).to(torch.uint8)
    # We only compute loss on masked tokens
    # TODO: Figure out why -1 does not work with xlm
    labels[~masked_indices] = -100
    

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (torch.bernoulli(torch.full(
        labels.shape, 0.8))).to(torch.uint8) & masked_indices
    tokens[indices_replaced] = mask_token_id 

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (torch.bernoulli(torch.full(
        labels.shape, 0.5))).to(torch.uint8) & masked_indices & invert(indices_replaced)

    tokens[indices_random] = torch.randint(
        len(tokenizer), (indices_random.sum(),), dtype=tokens.dtype)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels



