import torch
from pathlib import Path
from models.albert.CommentsDataset import CommentsDataset

torch.manual_seed(42)
ROOT_DIR = Path("/shared/0/projects/prosocial/data/finalized/")
cached_dev = torch.load(ROOT_DIR / 'cached.dev.full.albert.buffer')
downsampled_size = 1000000
indices = torch.randperm(len(cached_dev))[: downsampled_size]

subsampled_dev = CommentsDataset(cached_dev.meta_data[indices],
                                 cached_dev.subreddit_ids[indices],
                                 cached_dev.input_ids_tlc[indices],
                                 cached_dev.attention_mask_tlc[indices],
                                 cached_dev.input_ids_post[indices],
                                 cached_dev.attention_mask_post[indices],
                                 cached_dev.labels[indices])
# I/O
torch.save(subsampled_dev, ROOT_DIR / 'cached.dev.albert.buffer')
subsampled_dev.save_dataset(ROOT_DIR / 'cached.dev.albert.tensors.dict')
