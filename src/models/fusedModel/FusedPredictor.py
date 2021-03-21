import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict


class NonLinearBlock(nn.Module):
    """
    Pre-Activation NonLinear Block
    """
    def __init__(self, hidden_size, layer_norm_eps):
        super(NonLinearBlock, self).__init__()
        # hidden_size = 2 * feature_extractor.hidden_dimensions
        self.full_layer_layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        # self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=1)
        self.ffn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        out = self.full_layer_layer_norms(x)
        out = self.activation(out)
        out = self.ffn(out) + x
        return out


class FusedPredictor(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier: nn.Module,
                 **kwargs
                 ):
        super(FusedPredictor, self).__init__()
        self.feature_extractor = feature_extractor
        self.dropout = nn.Dropout(0.5)
        self.classifier = classifier

    @classmethod
    def from_scratch(cls,
                     model_constructor: nn.Module,
                     pretrained_feature_extractor_name_or_path: Path,
                     freeze_feature_extractor: bool,
                     **kwargs):
        # load checkpoints
        checkpoint = torch.load(pretrained_feature_extractor_name_or_path)
        state_dict = checkpoint['state_dict']
        model_key = kwargs.pop("model_key", "model_construct_params_dict")
        meta = {k: v for k, v in checkpoint.items() if k != 'state_dict'}

        # load model
        model = model_constructor(**meta[model_key])
        model.load_state_dict(state_dict)
        feature_extractor = model.as_feature_extractor(frozen=freeze_feature_extractor)

        classifier = nn.Linear(2 * feature_extractor.hidden_dimension, 2)  # number of labels is 2

        return cls(feature_extractor, classifier)

    @classmethod
    def from_pretrained(cls, feature_extractor_constructor: nn.Module, pretrained_system_name_or_path: Path):
        # load model
        checkpoint = torch.load(pretrained_system_name_or_path)
        state_dict = checkpoint['state_dict']
        feature_extractor_config = checkpoint['feature_extractor_config']
        # build skeleton
        feature_extractor = feature_extractor_constructor(**feature_extractor_config)
        classifier = nn.Linear(2 * feature_extractor.hidden_dimension, 2)  # number of labels is 2
        # load weights
        pretrained_model = cls(feature_extractor, classifier)
        pretrained_model.load_state_dict(state_dict)
        pretrained_model.eval()
        return pretrained_model

    def param_dict(self) -> OrderedDict:
        return OrderedDict({"feature_extractor.construct_param_dict": self.feature_extractor.construct_param_dict})

    def to_device(self, main_device: torch.device, embedding_device: torch.device, data_parallel: bool):
        self.feature_extractor.to_device(main_device, embedding_device, data_parallel)
        self.classifier.to(main_device)
        return self

    def forward(self,
                meta_data_l=None,
                subreddit_ids_l=None,
                input_ids_tlc_l=None,
                attention_mask_tlc_l=None,
                input_ids_post_l=None,
                attention_mask_post_l=None,
                meta_data_r=None,
                subreddit_ids_r=None,
                input_ids_tlc_r=None,
                attention_mask_tlc_r=None,
                input_ids_post_r=None,
                attention_mask_post_r=None,
                labels=None):
        left_features = self.feature_extractor(
                    meta_data=meta_data_l,
                    subreddit_ids=subreddit_ids_l,
                    input_ids_tlc=input_ids_tlc_l,  # top level comments
                    attention_mask_tlc=attention_mask_tlc_l,
                    input_ids_post=input_ids_post_l,  # post
                    attention_mask_post=attention_mask_post_l,
                )
        right_features = self.feature_extractor(
                    meta_data=meta_data_r,
                    subreddit_ids=subreddit_ids_r,
                    input_ids_tlc=input_ids_tlc_r,  # top level comments
                    attention_mask_tlc=attention_mask_tlc_r,
                    input_ids_post=input_ids_post_r,  # post
                    attention_mask_post=attention_mask_post_r,
                )
        hidden_reps = torch.cat((left_features, right_features), dim=1)
        logits = self.classifier(hidden_reps)

        outputs = (logits,) + (hidden_reps,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs
        return outputs
