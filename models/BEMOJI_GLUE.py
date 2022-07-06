"""
BEMOJI for Glue
:author: Qizhi Li
"""
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel

sys.path.append('..')
from static_data import file_path as fp


class PreprocessGlue:
    def __init__(self):
        self.AX_MNLI_label2id = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.AX_MNLI_id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.QNLI_RTE_label2id = {"entailment": 0, "not_entailment": 1}
        self.QNLI_RTE_id2label = {0: "entailment", 1: "not_entailment"}

    def CoLA(self, data_type):
        if data_type != 'test':
            data = pd.read_csv(os.path.join(fp.glue_data, 'CoLA/{}.tsv'.format(data_type)), header=None, delimiter='\t')
            x = data[3].tolist()
            y = data[1].tolist()

            return x, y
        else:
            data = pd.read_csv(os.path.join(fp.glue_data, 'CoLA/test.tsv'), delimiter='\t')
            x = data['sentence'].tolist()
            index = data['index'].tolist()

            return x, index

    def MNLI_m(self, data_type):

        if data_type != 'test_matched':
            with open(os.path.join(fp.glue_data, 'MNLI/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            premise, hypothesis, label = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                premise.append(split_contents[8])
                hypothesis.append(split_contents[9])
                label.append(self.AX_MNLI_label2id[split_contents[-1]])

            return premise, hypothesis, label
        else:
            with open(os.path.join(fp.glue_data, 'MNLI/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            premise, hypothesis, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                premise.append(split_contents[8])
                hypothesis.append(split_contents[9])
                index.append(split_contents[0])

            return premise, hypothesis, index

    def MNLI_mm(self, data_type):

        if data_type != 'test_mismatched':
            with open(os.path.join(fp.glue_data, 'MNLI/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            premise, hypothesis, label = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                premise.append(split_contents[8])
                hypothesis.append(split_contents[9])
                label.append(self.AX_MNLI_label2id[split_contents[-1]])

            return premise, hypothesis, label
        else:
            with open(os.path.join(fp.glue_data, 'MNLI/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            premise, hypothesis, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                premise.append(split_contents[8])
                hypothesis.append(split_contents[9])
                index.append(split_contents[0])

            return premise, hypothesis, index

    def MRPC(self, data_type):
        data = pd.read_csv(os.path.join(fp.glue_data, 'MRPC/{}.csv'.format(data_type)))
        text_a = data['sentence1'].tolist()
        text_b = data['sentence2'].tolist()
        label = data['label'].tolist()

        return text_a, text_b, label

    def QNLI(self, data_type):
        if data_type != 'test':
            with open(os.path.join(fp.glue_data, 'QNLI/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, label = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                label.append(self.QNLI_RTE_label2id[split_contents[-1]])

            return text_a, text_b, label
        else:
            with open(os.path.join(fp.glue_data, 'QNLI/test.tsv'), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                index.append(split_contents[0])

            return text_a, text_b, index

    def QQP(self, data_type):
        if data_type != 'test':
            data = pd.read_csv(os.path.join(fp.glue_data, 'QQP/{}.tsv'.format(data_type)), sep='\t', header=0)
            text_a = data['question1'].tolist()
            text_b = data['question2'].tolist()
            label = data['is_duplicate'].tolist()

            return text_a, text_b, label
        else:
            data = pd.read_csv(os.path.join(fp.glue_data, 'QQP/test.tsv'), sep='\t', header=0)
            text_a = data['question1'].tolist()
            text_b = data['question2'].tolist()
            index = data['id'].tolist()

            return text_a, text_b, index

    def RTE(self, data_type):
        if data_type != 'test':
            with open(os.path.join(fp.glue_data, 'RTE/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, label = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                label.append(self.QNLI_RTE_label2id[split_contents[-1]])
            return text_a, text_b, label

        else:
            with open(os.path.join(fp.glue_data, 'RTE/test.tsv'), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                index.append(split_contents[0])
            return text_a, text_b, index

    def SST2(self, data_type):
        if data_type != 'test':
            data = pd.read_csv(os.path.join(fp.glue_data, 'SST-2/{}.tsv'.format(data_type)), sep='\t', header=0)
            text = data['sentence'].tolist()
            label = data['label'].tolist()

            return text, label
        else:
            data = pd.read_csv(os.path.join(fp.glue_data, 'SST-2/test.tsv'), sep='\t', header=0)
            text = data['sentence'].tolist()
            index = data['index'].tolist()

            return text, index

    def STSB(self, data_type):
        if data_type != 'test':
            with open(os.path.join(fp.glue_data, 'STS-B/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, score = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[7])
                text_b.append(split_contents[8])
                score.append(float(split_contents[-1]))

            return text_a, text_b, score
        else:
            with open(os.path.join(fp.glue_data, 'STS-B/test.tsv'), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[7])
                text_b.append(split_contents[8])
                index.append(split_contents[0])

            return text_a, text_b, index

    def WNLI(self, data_type):
        if data_type != 'test':
            with open(os.path.join(fp.glue_data, 'WNLI/{}.tsv'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, label = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                label.append(int(split_contents[-1]))
            return text_a, text_b, label

        else:
            with open(os.path.join(fp.glue_data, 'WNLI/test.tsv'), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                index.append(split_contents[0])
            return text_a, text_b, index

    def AX(self, data_type):
        if data_type != 'test':
            with open(os.path.join(fp.glue_data, 'AX/multinli_1.0_{}.txt'.format(data_type)), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, label = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                try:
                    label.append(self.AX_MNLI_label2id[split_contents[0]])
                    text_a.append(split_contents[5])
                    text_b.append(split_contents[6])
                except KeyError:
                    continue
            return text_a, text_b, label

        else:
            with open(os.path.join(fp.glue_data, 'AX/AX.tsv'), 'r', encoding='utf-8') as f:
                contents = f.readlines()

            text_a, text_b, index = [], [], []
            for i, line in enumerate(contents):
                if i == 0:
                    continue
                split_contents = line.rstrip('\n').split('\t')
                text_a.append(split_contents[1])
                text_b.append(split_contents[2])
                index.append(split_contents[0])
            return text_a, text_b, index

class Config:
    def __init__(self, output_size, pre_train_epoch, parameter_path,
                 texts_max_length=64, batch_size=64, epochs=100):
        self.bert_path = '../static_data/bert-base-uncased'

        self.batch_size = batch_size
        self.lr = 1e-6
        self.weight_decay = 1e-4
        self.epochs = epochs

        self.embedding_size = 768
        self.output_size = output_size
        self.max_length = texts_max_length

        self.context_parameters_path = os.path.join(
            parameter_path, 'BEMOJI_github_context_ep_{}.bin'.format(pre_train_epoch))


class BEMOJICLS(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.config = config
        self.device = device

        tokenizer_class = BertTokenizer
        pretrained_weight = self.config.bert_path

        if args.dataset != 'STS-B':
            cls_class = BertForSequenceClassification
            self.bert_config = BertConfig.from_pretrained(pretrained_weight,
                                                          num_labels=self.config.output_size)
        else:
            cls_class = BertModel
            self.bert_config = BertConfig.from_pretrained(pretrained_weight)

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)

        self.bert = cls_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)

        self.dense = nn.Linear(config.embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

        context_state_dict = torch.load(config.context_parameters_path)
        # context_state_dict = torch.load(config.context_parameters_path, map_location='cpu')
        self.bert.load_state_dict(context_state_dict, strict=False)

    def forward(self, input_seqs1, input_seqs2=None):
        if input_seqs2 is None:
            context_tokens = self.tokenizer.batch_encode_plus(input_seqs1,
                                                              add_special_tokens=True,
                                                              max_length=self.config.max_length,
                                                              padding='max_length',
                                                              truncation='longest_first',
                                                              return_tensors='pt').to(self.device)
        else:
            context_tokens = self.tokenizer.batch_encode_plus(input_seqs1, input_seqs2,
                                                              max_length=self.config.max_length,
                                                              padding='max_length',
                                                              truncation='longest_first',
                                                              return_tensors='pt').to(self.device)

        if self.args.dataset == 'STS-B':
            # shape: [batch_size, embedding_size]
            context_cls = self.bert(**context_tokens).pooler_output
            output = self.sigmoid(self.dense(context_cls))
        else:
            output = self.bert(**context_tokens).logits

        return output
