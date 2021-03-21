import sys
sys.path.append('/home/jiajunb/prosocial-conversations')

from pathlib import Path
import torch
import logging
from models.roberta.RobertaDataSet import RobertaDataset

ROOT_DIR = Path("/shared/0/projects/prosocial/data/finalized/")

logger = logging.getLogger(__name__)

logger.info('Processing train dataset')
for mode in ['train', 'dev', 'test']:
    dataset = RobertaDataset.from_df(ROOT_DIR / f"{mode}_features.tsv",
                                     ROOT_DIR / f"{mode}_principle_values.tsv")
    torch.save(dataset, ROOT_DIR / f"cached.{mode}.roberta.buffer")
    dataset.save_dataset(ROOT_DIR / f"cached.{mode}.roberta.tensors.dict")
    del dataset

for mode in ['train', 'dev', 'test']:
    print('try loading from constructor')
    cached = torch.load(ROOT_DIR / f'cached.{mode}.roberta.buffer')

    print('try loading from dict')
    cached_dict = RobertaDataset.from_cached_dataset(ROOT_DIR / f'cached.{mode}.roberta.tensors.dict')


