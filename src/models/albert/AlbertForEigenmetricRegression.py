import logging
from torch import nn
import torch
from collections import OrderedDict
from transformers import AlbertModel
from src.models import GENERIC_ID
from pathlib import Path

logger = logging.getLogger(__name__)


class AlbertForEigenmetricRegression(nn.Module):
    def __init__(self,
                 num_labels: int,
                 top_comment_pretrained_model_name_or_path: Path,
                 post_pretrained_model_name_or_path: Path,
                 classifier_dropout_prob: float,
                 meta_data_size: int,
                 subreddit_pretrained_path: Path,
                 num_subreddit_embeddings: int,
                 subreddit_embeddings_size: int):
        super(AlbertForEigenmetricRegression, self).__init__()
        self.num_labels = num_labels
        self.construct_param_dict = \
            OrderedDict({"num_labels": num_labels,
                         "top_comment_pretrained_model_name_or_path": str(top_comment_pretrained_model_name_or_path),
                         "post_pretrained_model_name_or_path": str(post_pretrained_model_name_or_path),
                         "classifier_dropout_prob": classifier_dropout_prob,
                         "meta_data_size": meta_data_size,
                         "subreddit_pretrained_path": str(subreddit_pretrained_path),
                         "num_subreddit_embeddings": num_subreddit_embeddings,
                         "subreddit_embeddings_size": subreddit_embeddings_size})

        # load pretrained two albert models: one for top comment and the other for post
        self.top_comment_albert = AlbertModel.from_pretrained(top_comment_pretrained_model_name_or_path)
        self.post_albert = AlbertModel.from_pretrained(post_pretrained_model_name_or_path)
        # drop out layer
        self.dropout = nn.Dropout(classifier_dropout_prob)
        # the final classifier layer
        self.hidden_dimension = \
            self.top_comment_albert.config.hidden_size + self.post_albert.config.hidden_size \
            + meta_data_size + subreddit_embeddings_size
        self.classifier = nn.Linear(self.hidden_dimension, num_labels)
        self._init_weights(self.classifier)  # initialize the classifier
        # subreddit embeddings
        subreddit_embeddings = None
        if subreddit_pretrained_path:
            embeddings = torch.load(subreddit_pretrained_path)
            subreddit_embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False)
            logger.info(f"Loaded subreddit embeddings from {subreddit_pretrained_path}")
        elif num_subreddit_embeddings is not None and subreddit_embeddings_size is not None:
            subreddit_embeddings = nn.Embedding(num_subreddit_embeddings, subreddit_embeddings_size)
            # these numbers are gained from the pretrained embeddings
            torch.nn.init.normal_(subreddit_embeddings.weight, mean=0, std=0.49)
            logger.info(f"Initialized subreddit embeddings")
        self.subreddit_embeddings = subreddit_embeddings
        self.is_feature_extractor = False

    @classmethod
    def from_scratch(cls,
                     num_labels: int,
                     top_comment_pretrained_model_name_or_path: Path,
                     post_pretrained_model_name_or_path: Path,
                     classifier_dropout_prob: float,
                     meta_data_size: int,
                     subreddit_pretrained_path: Path,
                     num_subreddit_embeddings: int = None,
                     subreddit_embeddings_size: int = None):
        return cls(num_labels,
                   top_comment_pretrained_model_name_or_path,
                   post_pretrained_model_name_or_path,
                   classifier_dropout_prob,
                   meta_data_size,
                   subreddit_pretrained_path,
                   num_subreddit_embeddings,
                   subreddit_embeddings_size)

    @classmethod
    def from_pretrained(cls, pretrained_system_name_or_path: Path, model_key: str):
        checkpoint = torch.load(pretrained_system_name_or_path)
        state_dict = checkpoint['state_dict']
        meta = {k: v for k, v in checkpoint.items() if k != 'state_dict'}

        # load pretrained_model
        pretrained_model = cls(**meta[model_key])
        pretrained_model.load_state_dict(state_dict)
        pretrained_model.eval()
        return pretrained_model

    def param_dict(self) -> OrderedDict:
        return OrderedDict(self.construct_param_dict)

    def freeze_bert(self):
        for name, p in self.top_comment_albert.named_parameters():
            p.requires_grad = False

        for name, p in self.post_albert.named_parameters():
            p.requires_grad = False
        print("Freezed top_comment_albert and post_albert")
        return self

    @staticmethod
    def _init_weights(module: nn.Module):
        """
        initialize module weights
        the weights in the module are changed
        :param module: module to initialize
        :return: None
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        logger.info('Initialized weights')

    def to_device(self, main_device: torch.device, embedding_device: torch.device, data_parallel: bool):
        self.top_comment_albert = self.top_comment_albert.to(main_device)
        self.post_albert = self.post_albert.to(main_device)
        self.dropout = self.dropout.to(main_device)
        self.classifier = self.classifier.to(main_device)
        self.subreddit_embeddings = self.subreddit_embeddings.to(embedding_device)
        # DataParallel should be substituted with DistributedDataParallel in later versions
        if data_parallel and main_device != torch.device('cpu'):
            self.top_comment_albert = torch.nn.DataParallel(self.top_comment_albert)
            self.post_albert = torch.nn.DataParallel(self.post_albert)
            self.dropout = torch.nn.DataParallel(self.dropout)
            if not self.is_feature_extractor:
                self.classifier = torch.nn.DataParallel(self.classifier)
        if data_parallel and embedding_device != torch.device('cpu'):
            self.subreddit_embeddings = torch.nn.DataParallel(self.subreddit_embeddings)

        return self

    def as_feature_extractor(self, frozen: bool):
        self.is_feature_extractor = True
        print("Using AlbertForEigenmetricRegression as a feature extractor.")
        if frozen:
            for name, p in self.named_parameters():
                p.requires_grad = False
            print("All weights in model have been frozen.")
        return self

    def forward(
            self,
            meta_data=None,
            subreddit_ids=None,
            input_ids_tlc=None,  # top level comments
            attention_mask_tlc=None,
            token_type_ids_tlc=None,
            position_ids_tlc=None,
            head_mask_tlc=None,
            inputs_embeds_tlc=None,
            input_ids_post=None,  # post
            attention_mask_post=None,
            token_type_ids_post=None,
            position_ids_post=None,
            head_mask_post=None,
            inputs_embeds_post=None,
            labels=None,

    ):
        tlc_albert_repr = self.top_comment_albert(
            input_ids=input_ids_tlc,
            attention_mask=attention_mask_tlc,
            token_type_ids=token_type_ids_tlc,
            position_ids=position_ids_tlc,
            head_mask=head_mask_tlc,
            inputs_embeds=inputs_embeds_tlc,
        )[1]

        post_albert_repr = self.post_albert(
            input_ids=input_ids_post,
            attention_mask=attention_mask_post,
            token_type_ids=token_type_ids_post,
            position_ids=position_ids_post,
            head_mask=head_mask_post,
            inputs_embeds=inputs_embeds_post,
        )[1]

        # albert [CLS] representation:
        pooled_output = torch.cat((self.dropout(tlc_albert_repr), self.dropout(post_albert_repr)), dim=1)
        # 0.5 * community + 0.5 * generic
        alpha = 0.5
        subreddit_output = (alpha * self.subreddit_embeddings(subreddit_ids) + (1 - alpha) * self.subreddit_embeddings(
            torch.ones_like(subreddit_ids) * GENERIC_ID)).to(pooled_output.device)
        concatted_hidden = torch.cat((pooled_output, meta_data, subreddit_output), dim=1)

        if self.is_feature_extractor:  # dirty hack
            return concatted_hidden

        logits = self.classifier(concatted_hidden)

        # output logits; add hidden states and attention if they are here
        outputs = (logits,) + (tlc_albert_repr[2:],) + (post_albert_repr[2:],)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, tlc: ((hidden_states), (attentions)), post: ((hidden_states), (attentions))
