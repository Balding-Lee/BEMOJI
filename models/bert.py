import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer

class Config:
    def __init__(self, args, num_outputs, max_seq_length=64, batch_size=64):
        if args.dataset == 'chinese':
            self.epochs = 100
            self.lr = 1e-6
            if args.model == 'BERT_base':
                self.model_path = '../static_data/bert-base-chinese'
                if args.pretrain_epoch == 0:
                    self.save_path = '../models/model_parameters/bert_base_parameters.bin'
                else:
                    self.save_path = '../models/model_parameters/bert_base_fine_tune_parameters.bin'
            elif args.model == 'RoBERTa_base':
                self.model_path = '../static_data/chinese-roberta-wwm-ext'
                if args.pretrain_epoch == 0:
                    self.save_path = '../models/model_parameters/roberta_base_parameters.bin'
                else:
                    self.save_path = '../models/model_parameters/roberta_base_fine_tune_parameters.bin'
            elif args.model == 'RoBERTa_large':
                self.model_path = '../static_data/chinese-roberta-wwm-ext-large'
                if args.pretrain_epoch == 0:
                    self.save_path = '../models/model_parameters/roberta_large_parameters.bin'
                else:
                    self.save_path = '../models/model_parameters/roberta_large_fine_tune_parameters.bin'
            elif args.model == 'XLNet_base':
                self.model_path = '../static_data/chinese-xlnet-base'
                self.save_path = '../models/model_parameters/xlnet_base_parameters.bin'
            elif args.model == 'XLNet_mid':
                self.model_path = '../static_data/chinese-xlnet-mid'
                self.save_path = '../models/model_parameters/xlnet_mid_parameters.bin'
            elif args.model == 'DistilBert_base':
                self.model_path = '../static_data/distilbert-base-zh-cased'
                if args.pretrain_epoch == 0:
                    self.save_path = '../models/model_parameters/distilbert_base_parameters.bin'
                else:
                    self.save_path = '../models/model_parameters/distilbert_base_fine_tune_parameters.bin'
        else:
            self.epochs = 200
            self.lr = 1e-6
            if args.pretrain_epoch == 0:
                self.save_path = '../models/model_parameters/{}_{}_parameters.bin'.format(args.model, args.dataset)
            else:
                self.save_path = '../models/model_parameters/{}_{}_fine_tune_parameters.bin'.format(args.model,
                                                                                                    args.dataset)

            if args.model == 'BERT_base':
                self.model_path = '../static_data/bert-base-uncased'
            elif args.model == 'BERT_large':
                self.model_path = '../static_data/bert-large-uncased'
            elif args.model == 'RoBERTa_base':
                self.model_path = '../static_data/roberta-base'
            elif args.model == 'RoBERTa_large':
                self.model_path = '../static_data/roberta-large'
            elif args.model == 'DistilBert_base':
                self.model_path = '../static_data/distilbert-base-uncased'
            elif args.model == 'XLNet_base':
                self.model_path = '../static_data/xlnet-base-cased'
            elif args.model == 'XLNet_large':
                self.model_path = '../static_data/xlnet-large-cased'

        self.max_seq_length = max_seq_length
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.num_outputs = num_outputs


class BERT(nn.Module):
    def __init__(self, args, config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.common_config = config
        self.model_class, tokenizer_class, pretrained_weight = (AutoModelForSequenceClassification,
                                                                AutoTokenizer,
                                                                self.common_config.model_path)
        self.bert_config = AutoConfig.from_pretrained(pretrained_weight,
                                                      num_labels=self.common_config.num_outputs)
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weight)
        self.bert = self.model_class.from_pretrained(pretrained_weight, config=self.bert_config)

        if args.dataset == 'chinese':
            if args.pretrain_epoch != 0:
                parameter_path = os.path.join(
                    '../models/pretrain_parameters/', '{}_ep_{}.bin'.format(args.model, args.pretrain_epoch))
                context_state_dict = torch.load(parameter_path)
                self.bert.load_state_dict(context_state_dict, strict=False)
        else:
            if args.pretrain_epoch != 0:
                parameter_path = os.path.join(
                    '../models/pretrain_parameters/', '{}_github_ep_{}.bin'.format(args.model, args.pretrain_epoch))
                context_state_dict = torch.load(parameter_path)
                self.bert.load_state_dict(context_state_dict, strict=False)

        if args.fine_tune == 0:
            for name, parameter in self.bert.named_parameters():
                if name != 'classifier.weight' and name != 'classifier.bias':
                    parameter.requires_grad_ = False

    def forward(self, inputs):
        tokens = self.tokenizer.batch_encode_plus(inputs, add_special_tokens=True,
                                                  max_length=self.common_config.max_seq_length,
                                                  padding=True, truncation=True)
        input_ids = torch.tensor(tokens['input_ids']).to(self.device)
        attention_mask = torch.tensor(tokens['attention_mask']).to(self.device)
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        return outputs.logits
