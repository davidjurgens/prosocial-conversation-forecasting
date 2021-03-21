from pathlib import Path
import torch
import json
import logging
from models.albert.CommentsDataset import CommentsDataset
from models import ALBERT_EIGENMETRICS
import pandas as pd


ROOT_DIR = Path("/shared/0/projects/prosocial/data/finalized/")

logger = logging.getLogger(__name__)

with open(ROOT_DIR / 'subreddit_mappings.json', 'r') as ostream:
    subreddit_map = json.load(ostream)

for mode in ['train', 'dev', 'test']:
    logger.info('Processing test dataset')
    metrics_path = ROOT_DIR / f'{mode}_metrics.tsv'
    metrics_df = pd.read_csv(metrics_path, sep='\t')
    new_labels = torch.from_numpy(metrics_df[ALBERT_EIGENMETRICS].values).to(dtype=torch.float32)
    dataset, _ = CommentsDataset.from_old_dataset(ROOT_DIR / 'old' / f'cached.{mode}.albert.dict', labels=new_labels)
    torch.save(dataset, ROOT_DIR / f'cached.{mode}.albert.buffer')
    dataset.save_dataset(ROOT_DIR / f'cached.{mode}.albert.tensors.dict')
    logger.info(f'Saved the dataset at {ROOT_DIR}')
