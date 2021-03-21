import logging
import torch
import pandas as pd
from pathlib import Path
from src.models import BASELINE_ANNOTATION_MODEL_TLC_TEXT_HEADER, BASELINE_ANNOTATION_MODEL_LABEL_HEADER
from torch.utils.data import Dataset, TensorDataset
from transformers import AlbertTokenizer
import numpy as np

logger = logging.getLogger(__name__)


class BaselineDataset(Dataset):
    """
    dataset for the fused model
    (The dataframes fit in memory)
    """
    def __init__(self, dataset_module_l, dataset_module_r, labels):
        self.dataset_module_l = dataset_module_l
        self.dataset_module_r = dataset_module_r
        self.labels = labels

    @staticmethod
    def assemble_columns_headers(headers : list or str):
        if type(headers) == list:
            return [f'{x}_l' for x in headers], [f'{x}_r' for x in headers]
        elif type(headers) == str:
            return f'{headers}_l', f'{headers}_r'
        else:
            raise AssertionError(f'Undefined type: {type(headers)}')

    @classmethod
    def encode_text(cls, tlc_text: np.array, sequence_length):
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        instances_tlc = tokenizer.batch_encode_plus(tlc_text,
                                                    max_length=sequence_length,
                                                    pad_to_max_length=True, return_attention_masks=True,
                                                    return_token_type_ids=False)
        input_ids_tlc = torch.tensor(instances_tlc['input_ids'], dtype=torch.int32)
        attention_mask_tlc = torch.tensor(instances_tlc['attention_mask'], dtype=torch.int32)
        return input_ids_tlc, attention_mask_tlc

    @classmethod
    def from_df(cls, data_path: Path, subreddit_lookup: dict, sequence_length: int=128):
        logger.info("Reading dataframes")
        tlc_text_header_l, tlc_text_header_r = cls.assemble_columns_headers(BASELINE_ANNOTATION_MODEL_TLC_TEXT_HEADER)
        read_columns = [tlc_text_header_l] + [tlc_text_header_r]
        df = pd.read_csv(data_path, sep='\t', usecols=read_columns)
        # comment dataset
        tlc_text_l = df[tlc_text_header_l].values
        tlc_text_r = df[tlc_text_header_r].values
        input_ids_tlc_l, attention_mask_tlc_l = cls.encode_text(tlc_text_l, sequence_length)
        input_ids_tlc_r, attention_mask_tlc_r = cls.encode_text(tlc_text_r, sequence_length)
        labels = torch.from_numpy(df[BASELINE_ANNOTATION_MODEL_LABEL_HEADER].values).to(dtype=torch.long)
        dataset_module_l = TensorDataset(input_ids_tlc_l, attention_mask_tlc_l)
        dataset_module_r = TensorDataset(input_ids_tlc_r, attention_mask_tlc_r)
        return cls(dataset_module_l, dataset_module_r, labels)

    @classmethod
    def from_cached_dataset(cls, cached_path):
        data = torch.load(cached_path)
        return cls(**data)

    def save_dataset(self, cached_path):
        cached_path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save({
            'dataset_module_l': self.dataset_module_l,
            'dataset_module_r': self.dataset_module_r,
            'labels': self.labels,
        }, cached_path)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (
            *(self.dataset_module_l[idx]),
            *(self.dataset_module_r[idx]),
            self.labels[idx],
        )
