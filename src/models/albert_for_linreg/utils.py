import os
import logging
import pandas as pd
import numpy as np
from src.models import ALBERT_META_FEATURES
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def load_and_cache_dataset(args, tokenizer, split, comment=True, post=False):
    ds_cache = os.path.join(args.cache_dir, args.base_model + '_' +
        ('comment_' if comment else '') +
        ('post_' if post else '') + split + '.cache')

    if os.path.exists(ds_cache):
        logging.info("Loading cached dataset from " + ds_cache)
        return torch.load(ds_cache)
    logging.info("Start generating {} dataset".format(split))
    df_path = os.path.join(args.data_dir, "{}.tsv".format(split))
    
    features = pd.read_csv(df_path, sep='\t')

    ds_content = []
    if comment:
        comment_features = tokenizer.batch_encode_plus(
            features['Top_comment_text'].tolist(), 
            return_tensors='pt', return_attention_masks=True, 
            pad_to_max_length=True, return_token_type_ids=False,
            max_length=args.sequence_length)
        ds_content.append(comment_features['input_ids'])
        ds_content.append(comment_features['attention_mask'])

    if post:
        post_features = tokenizer.batch_encode_plus(
            features['Post_text'].tolist(), 
            return_tensors='pt', return_attention_masks=True, 
            pad_to_max_length=True, return_token_type_ids=False,
            max_length=args.sequence_length)
        ds_content.append(post_features['input_ids'])
        ds_content.append(post_features['attention_mask'])

    numerical_features = torch.tensor(features[ALBERT_META_FEATURES].to_numpy(),
     dtype=torch.float32)
    ds_content.append(numerical_features)

    pc_labels = torch.tensor(features['PC0'].to_numpy(), dtype=torch.float32)
    ds_content.append(pc_labels)
    dataset = TensorDataset(*ds_content)
    if os.path.isdir(args.cache_dir):
        logging.info("Saving cached dataset into " + ds_cache)
        torch.save(dataset, ds_cache)
    
    return dataset

    
def load_coeffs(args):
    if not args.linear_regression_coefs:
        return np.zeros(len(ALBERT_META_FEATURES))
    
    df = pd.read_csv(args.linear_regression_coefs)
    coeffs = df.iloc[0][ALBERT_META_FEATURES]
    return np.array(coeffs, dtype=np.float32)






    