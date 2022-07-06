import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel
from xml.dom.minidom import parse
import pyprind

sys.path.append('..')
from static_data import file_path as fp


class PreprocessDataset:
    def __init__(self):
        self.SemEval_label2id = {'neutral': 0, 'positive': 1, 'negative': 2}

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

    def ECISA(self, data_type):
        texts, labels, ids = [], [], []
        sentences_set = set()

        file_path = os.path.join(fp.chinese_sa_dataset,
                                 'SMP2019-ECISA/SMP2019_ECISA_{}.xml'.format(data_type))
        dom_tree = parse(file_path)
        root_node = dom_tree.documentElement
        docs = root_node.getElementsByTagName('Doc')
        for doc in docs:
            sentences = doc.getElementsByTagName('Sentence')
            doc_id = doc.getAttribute('ID')
            for sentence in sentences:
                # X: 文本数据, y: 标签 (0: neutral, 1: positive, 2: negative, 3: None),
                # words: list, 隐式情感词,
                # position: dict: {word: [pos]}, 隐式情感词语在句子中的位置

                if data_type != 'test':

                    label = sentence.getAttribute('label').rstrip('"')

                    if not label:
                        continue

                    text = sentence.childNodes[0].data

                    if text in sentences_set:
                        continue
                    else:
                        sentences_set.add(text)
                        texts.append(text)
                        labels.append(int(label))

                else:
                    text = sentence.childNodes[0].data
                    texts.append(text)
                    ids.append('{}-{}'.format(doc_id, sentence.getAttribute('ID')))

        if data_type == 'test':
            return texts, ids
        else:
            return texts, labels

    def SemEval(self, data_type):
        texts, labels = [], []
        file_path = os.path.join(fp.english_data, 'SemEval/{}.tsv'.format(data_type))
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = f.readlines()

        for line in contents:
            split_line = line.rstrip('\n').split('\t')
            text = split_line[3]
            label = split_line[2]

            if label == 'objective-OR-neutral' or label == 'objective':
                continue

            texts.append(text)
            labels.append(self.SemEval_label2id[label])

        return texts, labels

    def NLPCC(self, data_type):
        df = pd.read_csv(os.path.join(fp.chinese_sa_dataset, 'NLPCC2014/{}.csv'.format(data_type)))
        texts = df['texts'].tolist()
        labels = df['labels'].tolist()

        return texts, labels


class Config:
    def __init__(self, args, output_size,
                 texts_max_length=64, batch_size=64, epochs=100):
        if args.dataset == 'SST-2' or args.dataset == 'SemEval':
            self.bert_path = '../static_data/bert-base-uncased'
        else:
            self.bert_path = '../static_data/bert-base-chinese'

        self.batch_size = batch_size
        self.lr = 1e-6
        self.weight_decay = 1e-4
        self.epochs = epochs

        self.embedding_size = 768
        self.output_size = output_size
        self.max_length = texts_max_length


class BERTCLS(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.config = config
        self.device = device

        tokenizer_class = BertTokenizer
        pretrained_weight = self.config.bert_path

        cls_class = BertForSequenceClassification
        self.bert_config = BertConfig.from_pretrained(pretrained_weight,
                                                      num_labels=self.config.output_size)

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)

        self.bert = cls_class.from_pretrained(pretrained_weight, config=self.bert_config).to(self.device)

    def forward(self, input_seqs):
        context_tokens = self.tokenizer.batch_encode_plus(input_seqs,
                                                          add_special_tokens=True,
                                                          max_length=self.config.max_length,
                                                          padding='max_length',
                                                          truncation='longest_first',
                                                          return_tensors='pt').to(self.device)

        output = self.bert(**context_tokens).logits

        return output