from pathlib import Path
import torch
import json
import logging
from models.albert.CommentsDataset import CommentsDataset

ROOT_DIR = Path("/shared/0/projects/prosocial/data/finalized/dataframes")
OUTPUT_DIR = Path("/shared/0/projects/prosocial/data/finalized/data_cache/albert")

logger = logging.getLogger(__name__)

with open(ROOT_DIR / 'subreddit_mappings.json', 'r') as istream:
    subreddit_map = json.load(istream)

logger.info('Processing train dataset')
dataset, _ = CommentsDataset.from_df(data_path=ROOT_DIR / "train.tsv",
                                     subreddit_lookup=subreddit_map)

torch.save(dataset, OUTPUT_DIR / 'cached.train.albert.buffer')
dataset.save_dataset(OUTPUT_DIR / 'cached.train.albert.tensors.dict')

