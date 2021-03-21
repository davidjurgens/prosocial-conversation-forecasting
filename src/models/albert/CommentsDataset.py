import logging
import torch
import pandas as pd

from src.models import ALBERT_TOP_COMMENT_TEXT_HEADER, ALBERT_POST_TEXT_HEADER
from src.models import GENERIC_TOKEN, GENERIC_ID
from src.models import ALBERT_META_FEATURES, ALBERT_SUBREDDIT_HEADER, ALBERT_EIGENMETRICS
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

logger = logging.getLogger(__name__)


class CommentsDataset(Dataset):
    """
    Comments dataset
    (The dataframes fit in memory)
    """

    def __init__(self,
                 meta_data,
                 subreddit_ids,
                 input_ids_tlc,
                 attention_mask_tlc,
                 input_ids_post,
                 attention_mask_post,
                 labels=None):
        self.meta_data = meta_data
        self.subreddit_ids = subreddit_ids
        self.input_ids_tlc = input_ids_tlc
        self.attention_mask_tlc = attention_mask_tlc
        self.input_ids_post = input_ids_post
        self.attention_mask_post = attention_mask_post
        self.labels = labels

    @classmethod
    def from_df(cls, data_path, subreddit_lookup=None, sequence_length=128):
        logger.info("Reading dataframes")
        df = pd.read_csv(data_path, sep='\t')

        subreddit_names = df[ALBERT_SUBREDDIT_HEADER].values
        if subreddit_lookup is None:
            # encode subreddit
            logger.info("Encoding subreddits")
            subreddit_primary_names = sorted(list(set(subreddit_names)))
            subreddit_lookup = dict(zip(subreddit_primary_names, range(1, len(subreddit_primary_names) + 1)))
            subreddit_lookup[GENERIC_TOKEN] = GENERIC_ID

        meta_data, subreddit_ids, \
            input_ids_tlc, attention_mask_tlc, \
            input_ids_post, attention_mask_post = cls.encode_features(df,
                                                                      subreddit_lookup,
                                                                      ALBERT_TOP_COMMENT_TEXT_HEADER,
                                                                      ALBERT_POST_TEXT_HEADER,
                                                                      ALBERT_META_FEATURES,
                                                                      ALBERT_SUBREDDIT_HEADER,
                                                                      sequence_length)
        # label
        labels = torch.from_numpy(df[ALBERT_EIGENMETRICS].values).to(dtype=torch.float32)

        return cls(meta_data,
                   subreddit_ids,
                   input_ids_tlc,
                   attention_mask_tlc,
                   input_ids_post,
                   attention_mask_post,
                   labels), subreddit_lookup

    @staticmethod
    def encode_features(df: pd.DataFrame,
                        subreddit_lookup: dict,
                        tlc_text_header: str,
                        post_text_header: str,
                        meta_feature_headers: list,
                        subreddit_header: str,
                        sequence_length: int):
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        tlc_text = df[tlc_text_header].values
        post_text = df[post_text_header].values
        # top level text encodings
        logger.info("Tokenizing top comments")
        instances_tlc = tokenizer.batch_encode_plus(tlc_text,
                                                    max_length=sequence_length,
                                                    pad_to_max_length=True, return_attention_masks=True,
                                                    return_token_type_ids=False)
        input_ids_tlc = torch.tensor(instances_tlc['input_ids'], dtype=torch.int32)
        attention_mask_tlc = torch.tensor(instances_tlc['attention_mask'], dtype=torch.int32)
        # post text encodings
        logger.info("Tokenizing posts")
        instances_post = tokenizer.batch_encode_plus(post_text,
                                                     max_length=sequence_length,
                                                     pad_to_max_length=True, return_attention_masks=True,
                                                     return_token_type_ids=False)
        input_ids_post = torch.tensor(instances_post['input_ids'], dtype=torch.int32)
        attention_mask_post = torch.tensor(instances_post['attention_mask'], dtype=torch.int32)

        # meta data
        meta_data = torch.from_numpy(df[meta_feature_headers].values).to(dtype=torch.float32)

        subreddit_names = df[subreddit_header].values
        subreddit_ids = torch.tensor(
            list(map(
                lambda name: subreddit_lookup[name] if name in subreddit_lookup else subreddit_lookup[GENERIC_TOKEN],
                subreddit_names)), dtype=torch.int32)

        return meta_data, subreddit_ids, input_ids_tlc, attention_mask_tlc, input_ids_post, attention_mask_post

    @classmethod
    def from_cached_dataset(cls, cached_path):
        data = torch.load(cached_path)
        return cls(**data)

    @classmethod
    def from_old_dataset(cls, cached_path, **kwargs):
        data = torch.load(cached_path)
        meta_data = kwargs.pop("meta_data", data["meta_data"])
        subreddit_ids = kwargs.pop("subreddit_ids", data["subreddit_ids"])
        input_ids_tlc = kwargs.pop("input_ids_tlc", data["input_ids_tlc"])
        attention_mask_tlc = kwargs.pop("attention_mask_tlc", data["attention_mask_tlc"])
        input_ids_post = kwargs.pop("input_ids_posts", data["input_ids_posts"])
        attention_mask_post = kwargs.pop("attention_mask_post", data["attention_mask_post"])
        labels = kwargs.pop("labels", data["labels"])
        return cls(meta_data,
                   subreddit_ids,
                   input_ids_tlc,
                   attention_mask_tlc,
                   input_ids_post,
                   attention_mask_post,
                   labels)

    def get_state_dict(self):
        """
        get the "pointer" to / a shallow copy of the underlying data
        :return: 
        """
        return {
            "meta_data": self.meta_data,
            "subreddit_ids": self.subreddit_ids,
            "input_ids_tlc": self.input_ids_tlc,
            "attention_mask_tlc": self.attention_mask_tlc,
            "input_ids_post": self.input_ids_post,
            "attention_mask_post": self.attention_mask_post,
            "labels": self.labels,
        }

    def save_dataset(self, cached_path):
        cached_path.parents[0].mkdir(parents=True, exist_ok=True)
        torch.save(self.get_state_dict(), cached_path)

    def __len__(self):
        return self.meta_data.shape[0]

    def __getitem__(self, idx):
        slices = (self.meta_data[idx],
                  self.subreddit_ids[idx],
                  self.input_ids_tlc[idx],
                  self.attention_mask_tlc[idx],
                  self.input_ids_post[idx],
                  self.attention_mask_post[idx])
        if self.labels is not None:
            slices = slices + (self.labels[idx],)
        return slices
