"""
BEMOJI 模型实现 CLS
:author: Qizhi Li
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer, BertModel, BertForSequenceClassification

from transformers.models.bert.modeling_bert import BertModel


class Config:
    def __init__(self, args, output_size, pre_train_epoch, parameter_path,
                 texts_max_length=64, emoji_max_length=64, batch_size=64,
                 epochs=100):
        self.bert_chinese_path = '../static_data/bert-base-chinese'
        self.bert_english_path = '../static_data/bert-base-uncased'

        self.batch_size = batch_size
        self.lr = 1e-6
        self.weight_decay = 1e-4
        self.epochs = epochs

        self.embedding_size = 768
        self.fusion_layer_size = 768
        self.output_size = output_size
        # self.chinese_texts_max_length = 64
        # self.english_texts_max_length = 64
        # self.chinese_emoji_max_length = 32
        # self.english_emoji_max_length = 32
        self.texts_max_length = texts_max_length
        self.emoji_max_length = emoji_max_length

        if args.dataset == 'chinese':
            self.context_parameters_path = os.path.join(
                parameter_path, 'BEMOJI_context_ep_{}.bin'.format(pre_train_epoch))
            self.emoji_parameters_path = os.path.join(
                parameter_path, 'BEMOJI_emoji_ep_{}.bin'.format(pre_train_epoch))
        else:
            self.context_parameters_path = os.path.join(
                parameter_path, 'BEMOJI_github_context_ep_{}.bin'.format(pre_train_epoch))
            self.emoji_parameters_path = os.path.join(
                parameter_path, 'BEMOJI_github_emoji_ep_{}.bin'.format(pre_train_epoch))
            # self.emoji_parameters_path = os.path.join(
            #     parameter_path, 'BEMOJI_github_emoji_ep_{}.bin'.format(fine_tune_epoch))


class BEMOJICLS(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.config = config
        self.device = device

        tokenizer_class = BertTokenizer

        cls_class = BertModel

        if args.dataset == 'chinese':
            pretrained_weight = self.config.bert_chinese_path
        else:
            pretrained_weight = self.config.bert_english_path

        self.bert_config = BertConfig.from_pretrained(pretrained_weight)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)

        self.context_bert = cls_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)
        self.emoji_bert = cls_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)

        self.context_dense = nn.Linear(config.embedding_size,
                                       config.fusion_layer_size,
                                       bias=False)
        self.emoji_dense = nn.Linear(config.embedding_size,
                                     config.fusion_layer_size,
                                     bias=False)
        self.bias = nn.Parameter(torch.zeros(config.fusion_layer_size, ))
        self.activate = nn.GELU()

        if args.ablation == 'wf':
            self.fc = nn.Linear(config.fusion_layer_size * 2, config.output_size)
        else:
            self.fc = nn.Linear(config.fusion_layer_size, config.output_size)
        self.softmax = nn.Softmax()

        if args.pre_train_epoch != 0:
            context_state_dict = torch.load(config.context_parameters_path)
            emoji_state_dict = torch.load(config.emoji_parameters_path)
            self.context_bert.load_state_dict(context_state_dict, strict=False)
            self.emoji_bert.load_state_dict(emoji_state_dict, strict=False)

        if args.fine_tune == 0:
            for name, parameter in self.context_bert.named_parameters():
                parameter.requires_grad_ = False

            for name, parameter in self.emoji_bert.named_parameters():
                parameter.requires_grad_ = False

    def forward(self, input_seqs, input_emojis=None):
        context_tokens = self.tokenizer.batch_encode_plus(input_seqs,
                                                          add_special_tokens=True,
                                                          max_length=self.config.texts_max_length,
                                                          padding='max_length',
                                                          truncation='longest_first')

        context_ids = torch.tensor(context_tokens['input_ids']).to(self.device)
        context_att_mask = torch.tensor(context_tokens['attention_mask']).to(self.device)
        # shape: [batch_size, embedding_size]
        context_cls = self.context_bert(context_ids, attention_mask=context_att_mask).pooler_output

        if self.args.ablation != 'we':
            # ablation == 'None' or ablation == 'wf'
            if self.args.dataset == 'chinese':
                emojis_cls = []
                # input_emojis: shape: (batch, emojis_defs)
                # [[def_11], [def_21, def_22], [def_31], ...]
                for emoji_defs in input_emojis:
                    emoji_tokens = self.tokenizer.batch_encode_plus(emoji_defs,
                                                                    add_special_tokens=True,
                                                                    max_length=self.config.emoji_max_length,
                                                                    padding='max_length',
                                                                    truncation='longest_first')
                    emoji_ids = torch.tensor(emoji_tokens['input_ids']).to(self.device)
                    emoji_att_mask = torch.tensor(emoji_tokens['attention_mask']).to(self.device)
                    # shape: [num_emojis, embedding_size]
                    emoji_cls = self.emoji_bert(emoji_ids, attention_mask=emoji_att_mask).pooler_output
                    # emojis_cls.append(self.global_max_1dpool(emoji_cls))
                    emojis_cls.append(emoji_cls.max(dim=0).values.unsqueeze(dim=0))
                # shape: [batch, embedding_size]
                emojis_cls = torch.cat(emojis_cls, 0)
            else:
                emoji_tokens = self.tokenizer.batch_encode_plus(input_emojis,
                                                                add_special_tokens=True,
                                                                max_length=self.config.emoji_max_length,
                                                                padding='max_length',
                                                                truncation='longest_first')
                emoji_ids = torch.tensor(emoji_tokens['input_ids']).to(self.device)
                emoji_att_mask = torch.tensor(emoji_tokens['attention_mask']).to(self.device)
                # shape: [batch, embedding_size]
                emojis_cls = self.emoji_bert(emoji_ids, attention_mask=emoji_att_mask).pooler_output

            if self.args.ablation != 'wf':
                # ablation == 'None'
                fusion_hidden_state = self.activate(
                    self.context_dense(context_cls) + self.emoji_dense(emojis_cls) + self.bias)
            else:
                # ablation == 'wf'
                # shape: [batch, embedding_size * 2]
                fusion_hidden_state = torch.cat((context_cls, emojis_cls), 1)
        else:
            # ablation == 'we'
            # shape: [batch, embedding_size]
            fusion_hidden_state = self.activate(self.context_dense(context_cls) + self.bias)

        output = self.softmax(self.fc(fusion_hidden_state))

        return output

        # if self.args.dataset == 'chinese':
        #     context_cls = self.context_bert(context_ids, attention_mask=context_att_mask).pooler_output
        #
        #     emojis_cls = []
        #     # input_emojis: shape: (batch, emojis_defs)
        #     # [[def_11], [def_21, def_22], [def_31], ...]
        #     for emoji_defs in input_emojis:
        #
        #         emoji_tokens = self.tokenizer.batch_encode_plus(emoji_defs,
        #                                                         add_special_tokens=True,
        #                                                         max_length=self.config.emoji_max_length,
        #                                                         padding='max_length',
        #                                                         truncation='longest_first')
        #         emoji_ids = torch.tensor(emoji_tokens['input_ids']).to(self.device)
        #         emoji_att_mask = torch.tensor(emoji_tokens['attention_mask']).to(self.device)
        #         # shape: [num_emojis, embedding_size]
        #         emoji_cls = self.emoji_bert(emoji_ids, attention_mask=emoji_att_mask).pooler_output
        #         # emojis_cls.append(self.global_max_1dpool(emoji_cls))
        #         emojis_cls.append(emoji_cls.max(dim=0).values.unsqueeze(dim=0))
        #
        #     emojis_cls = torch.cat(emojis_cls, 0)
        #
        #     fusion_hidden_state = self.activate(
        #         self.context_dense(context_cls) + self.emoji_dense(emojis_cls) + self.bias)
        #
        #     output = self.softmax(self.fc(fusion_hidden_state))
        #
        #     return output
        #
        # else:
        #     context_cls = self.context_bert(context_ids, attention_mask=context_att_mask).pooler_output
        #     fusion_hidden_state = self.activate(
        #         self.context_dense(context_cls) + self.bias)
        #
        #     output = self.softmax(self.fc(fusion_hidden_state))
        #
        #     return output
