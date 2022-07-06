"""
case study
:author: Qizhi Li
"""
import argparse
import sys
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertModel, BertForMaskedLM
# from bertviz import model_view

sys.path.append('..')
import utils
from static_data import file_path as fp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--dataset',
                    help='pretrained parameter type, Chinese, or English',
                    default='Chinese')
parser.add_argument('--k',
                    help='Top k',
                    type=int,
                    default=5)
parser.add_argument('-m',
                    '--model',
                    help='BEMOJI, BERT',
                    default='BEMOJI')
parser.add_argument('-t',
                    '--text',
                    help='The text that you want to input into the model',
                    default='你站在桥上看风景，看风景的人在楼上看你')
args = parser.parse_args()

emoji_max_length = 64
prefix = '[MASK]'
if args.dataset == 'Chinese':
    text = ['{}{}'.format(prefix, args.text)]
else:
    text = ['{} {}'.format(prefix, args.text)]

if args.dataset == 'Chinese':
    bert_path = '../static_data/bert-base-chinese'
    if args.model == 'BEMOJI':
        context_pretrain_weight = os.path.join(fp.pre_train_parameters, 'BEMOJI_context_ep_9.bin')
    else:
        context_pretrain_weight = os.path.join(fp.pre_train_parameters, 'BERT_base_ep_5.bin')
    emoji_pretrain_weight = os.path.join(fp.pre_train_parameters, 'BEMOJI_emoji_ep_9.bin')

    emoji2def = pd.read_csv(fp.weibo_emoji2def)
    emoji_list = list(emoji2def['emoji 表情'])
    emojis, defs = [], []
    for _, line in emoji2def.iterrows():
        emojis.append(line['微博表情'])
        defs.append(line['表情定义'])
else:
    bert_path = '../static_data/bert-base-uncased'
    if args.model == 'BEMOJI':
        context_pretrain_weight = os.path.join(fp.pre_train_parameters, 'BEMOJI_github_context_ep_1.bin')
    else:
        context_pretrain_weight = os.path.join(fp.pre_train_parameters, 'BERT_base_github_ep_1.bin')
    emoji_pretrain_weight = os.path.join(fp.pre_train_parameters, 'BEMOJI_github_emoji_ep_1.bin')
    # text = ['{} I really love coding'.format(prefix)]

    emoji2def = pd.read_csv(fp.github_emoji2def)
    emoji_list = list(emoji2def['emoji'])
    emojis, defs = [], []
    for _, line in emoji2def.iterrows():
        emojis.append(line['emoji'])
        defs.append(line['definition'])

tokenizer_class = BertTokenizer
bert_class, mlm_class = BertModel, BertForMaskedLM

bert_config = BertConfig.from_pretrained(bert_path)
tokenizer = tokenizer_class.from_pretrained(bert_path)
context_bert = mlm_class.from_pretrained(bert_path, config=bert_config).to(device)
emoji_bert = bert_class.from_pretrained(bert_path, config=bert_config).to(device)

if args.model == 'BEMOJI':
    context_state_dict = torch.load(context_pretrain_weight)
    context_bert.load_state_dict(context_state_dict, strict=False)

emoji_state_dict = torch.load(emoji_pretrain_weight)
emoji_bert.load_state_dict(emoji_state_dict, strict=False)

if args.dataset == 'Chinese':
    if not os.path.exists(os.path.join(fp.case_study, 'chinese_emoji_representation.pkl')):
        with torch.no_grad():
            emoji_inputs = tokenizer.batch_encode_plus(defs,
                                                       add_special_tokens=True,
                                                       max_length=emoji_max_length,
                                                       padding='max_length',
                                                       truncation='longest_first')

            emoji_ids = torch.tensor(emoji_inputs['input_ids']).to(device)
            emoji_att_mask = torch.tensor(emoji_inputs['attention_mask']).to(device)
            # shape: [batch_size, embedding_size]
            emoji_pooler_output = emoji_bert(emoji_ids, attention_mask=emoji_att_mask).pooler_output

        utils.write_file('pkl', os.path.join(fp.case_study, 'chinese_emoji_representation.pkl'), emoji_pooler_output)
    else:
        emoji_pooler_output = utils.read_file('pkl', os.path.join(fp.case_study, 'chinese_emoji_representation.pkl'))
else:
    if not os.path.exists(os.path.join(fp.case_study, 'english_emoji_representation.pkl')):
        with torch.no_grad():
            emoji_inputs = tokenizer.batch_encode_plus(defs,
                                                       add_special_tokens=True,
                                                       max_length=emoji_max_length,
                                                       padding='max_length',
                                                       truncation='longest_first')

            emoji_ids = torch.tensor(emoji_inputs['input_ids']).to(device)
            emoji_att_mask = torch.tensor(emoji_inputs['attention_mask']).to(device)
            # shape: [batch_size, embedding_size]
            emoji_pooler_output = emoji_bert(emoji_ids, attention_mask=emoji_att_mask).pooler_output

        utils.write_file('pkl', os.path.join(fp.case_study, 'english_emoji_representation.pkl'), emoji_pooler_output)
    else:
        # shape: [64, 768]
        emoji_pooler_output = utils.read_file('pkl', os.path.join(fp.case_study, 'english_emoji_representation.pkl'))

with torch.no_grad():
    mse_loss = torch.nn.MSELoss(reduce=False)
    inputs = tokenizer.batch_encode_plus(text, return_tensors='pt').to(device)
    # shape: [1, 768]
    # context_pooler_output = context_bert(**inputs, output_hidden_states=True).hidden_states[12][:, 1, :]
    output = context_bert(**inputs, output_hidden_states=True)

    context_pooler_output = output.hidden_states[12][:, 1, :]

    # shape: [64, 768]
    context_representation = context_pooler_output.repeat(len(emoji_pooler_output), 1)
    loss = mse_loss(context_representation, emoji_pooler_output)
    loss_mean = torch.mean(loss, dim=1)
    topk_indices = loss_mean.topk(args.k, largest=False)[1].tolist()

    print(text[0].lstrip(prefix))
    for i in topk_indices:
        print(emojis[i], end=' ')
    print('\n')
