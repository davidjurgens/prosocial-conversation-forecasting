from __future__ import absolute_import, division, print_function

import torch
from torch import nn as nn
from transformers import AutoModel
from src.models import ALBERT_META_FEATURES

class BertWrapper(nn.Module):
    def __init__(self, args):
        super(BertWrapper, self).__init__()

        self.model_architecture = args.model_architecture

        self.bert_comment = AutoModel.from_pretrained(args.comment_model,
         cache_dir=args.cache_dir)
        self.output_dim = self.bert_comment.config.hidden_size

        if args.model_architecture == 'comment_post':
            self.bert_post = AutoModel.from_pretrained(args.post_model,
             cache_dir=args.cache_dir)
            self.output_dim += self.bert_comment.config.hidden_size

    def forward(self, batch):
        if self.model_architecture == 'comment':
            iid0, attn0 = batch
            CLS = self.bert_comment(input_ids=iid0, attention_mask=attn0)[0][:,0,:]
        elif self.model_architecture == 'comment_post':
            iid0, attn0, iid1, attn1 = batch

            CLS0 = self.bert_comment(input_ids=iid0, attention_mask=attn0)[0][:,0,:]
            CLS1 = self.bert_post(input_ids=iid1, attention_mask=attn1)[0][:,0,:]

            CLS = torch.cat((CLS0, CLS1), dim=1)
        else:
            raise NotImplementedError
        return CLS




class LinRegModel(nn.Module):
    def __init__(self, coeffs, args):
        super(LinRegModel, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        LINEAR_FEATURES = len(ALBERT_META_FEATURES)
        self.linear = torch.nn.Linear(LINEAR_FEATURES, 1)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        self.dropout = nn.Dropout(0.25)
        self.linear.weight.copy_(torch.Tensor(coeffs))


        self.bert = BertWrapper(args)

        self.linear_comb = torch.nn.Linear(self.bert.output_dim + LINEAR_FEATURES, 128)
        self.actv = nn.PReLU()
        self.output = torch.nn.Linear(128, 1)

    def forward(self, batch):
        meta_features = batch[-2]
        labels = batch[-1]
        text_hidden = self.bert(batch[:-2])
        concat_feature = torch.cat((text_hidden, meta_features), dim=1)
        concat_feature = self.dropout(concat_feature)
        output = self.output(self.actv(self.linear_comb(concat_feature))) + self.linear(meta_features)

        return self.loss(output.view(-1), labels.view(-1)), output