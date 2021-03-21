import logging
import torch
import pandas as pd
from pathlib import Path
from src.models import ANNOTATION_ALBERT_TLC_TEXT_HEADER, ANNOTATION_ALBERT_POST_TEXT_HEADER
from src.models import ANNOTATION_ALBERT_META_FEATURES, ANNOTATION_ALBERT_SUBREDDIT_HEADER
from src.models import GENERIC_TOKEN, GENERIC_ID, ANNOTATION_ALBERT_LABEL_HEADER
from src.models.albert.CommentsDataset import CommentsDataset
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

logger = logging.getLogger(__name__)


class FusedDataset(Dataset):
    """
    dataset for the fused model
    (The dataframes fit in memory)
    """

    def __init__(self, comment_dataset_l, comment_dataset_r, labels):
        self.comment_dataset_l = comment_dataset_l
        self.comment_dataset_r = comment_dataset_r
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
    def from_df(cls, data_path: Path, subreddit_lookup: dict, sequence_length: int=128):
        feature_columns = [ANNOTATION_ALBERT_SUBREDDIT_HEADER] + ANNOTATION_ALBERT_META_FEATURES \
                          + [ANNOTATION_ALBERT_TLC_TEXT_HEADER] + [ANNOTATION_ALBERT_POST_TEXT_HEADER]
        features_l, features_r = cls.assemble_columns_headers(feature_columns)
        read_columns = features_l + features_r + [ANNOTATION_ALBERT_LABEL_HEADER]
        logger.info("Reading dataframes")
        df = pd.read_csv(data_path, sep='\t', usecols=read_columns)
        # comment dataset
        tlc_text_header_l, tlc_text_header_r = cls.assemble_columns_headers(ANNOTATION_ALBERT_TLC_TEXT_HEADER)
        post_text_header_l, post_text_header_r = cls.assemble_columns_headers(ANNOTATION_ALBERT_POST_TEXT_HEADER)
        meta_feature_headers_l, meta_feature_headers_r = cls.assemble_columns_headers(ANNOTATION_ALBERT_META_FEATURES)
        subreddit_header_l, subreddit_header_r = cls.assemble_columns_headers(ANNOTATION_ALBERT_SUBREDDIT_HEADER)
        logger.info("Tokenizing top comments")
        comment_dataset_l = CommentsDataset(
            *CommentsDataset.encode_features(df=df[features_l],
                                             subreddit_lookup=subreddit_lookup,
                                             tlc_text_header=tlc_text_header_l,
                                             post_text_header=post_text_header_l,
                                             meta_feature_headers=meta_feature_headers_l,
                                             subreddit_header=subreddit_header_l,
                                             sequence_length=sequence_length)
        )
        comment_dataset_r = CommentsDataset(
            *CommentsDataset.encode_features(df=df[features_r],
                                             subreddit_lookup=subreddit_lookup,
                                             tlc_text_header=tlc_text_header_r,
                                             post_text_header=post_text_header_r,
                                             meta_feature_headers=meta_feature_headers_r,
                                             subreddit_header=subreddit_header_r,
                                             sequence_length=sequence_length)
        )
        labels = torch.from_numpy(df[ANNOTATION_ALBERT_LABEL_HEADER].values).to(dtype=torch.long)
        return cls(comment_dataset_l, comment_dataset_r, labels)

    @classmethod
    def from_cached_dataset(cls, cached_path):
        data = torch.load(cached_path)
        comment_dataset_l = CommentsDataset(**data['comment_dataset_l'])
        comment_dataset_r = CommentsDataset(**data['comment_dataset_r'])
        labels = data['labels']
        return cls(comment_dataset_l, comment_dataset_r, labels)

    def save_dataset(self, cached_path):
        cached_path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save({
            'comment_dataset_l': self.comment_dataset_l.get_state_dict(),
            'comment_dataset_r': self.comment_dataset_r.get_state_dict(),
            'labels': self.labels,
        }, cached_path)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (
            *(self.comment_dataset_l[idx][:6]),
            *(self.comment_dataset_r[idx][:6]),
            self.labels[idx],
        )
