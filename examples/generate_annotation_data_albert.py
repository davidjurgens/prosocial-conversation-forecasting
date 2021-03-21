from pathlib import Path
import torch
import json
import logging
from models.fusedModel.FusedDataset import FusedDataset

version = "finalized"
ROOT_DIR = Path(f"/shared/0/projects/prosocial/data/{version}/")
OUTPUT_DIR = Path(f"/shared/0/projects/prosocial/data/{version}/data_cache/fused_albert")

logger = logging.getLogger(__name__)

with open(ROOT_DIR / 'dataframes/subreddit_mappings.json', 'r') as istream:
    subreddit_map = json.load(istream)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for mode in ['train', 'dev', 'test']:
    logger.info(f'Processing {mode} dataset')
    dataset = FusedDataset.from_df(data_path=(ROOT_DIR / 'annotation_data_splits' / f'{mode}.tsv'),
                                   subreddit_lookup=subreddit_map)
    torch.save(dataset, OUTPUT_DIR / f'cached.{mode}.annotation.albert.buffer')
    dataset.save_dataset(OUTPUT_DIR / f'cached.{mode}.annotation.albert.tensors.dict')


