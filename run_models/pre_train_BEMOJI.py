"""
BEMOJI for pre-training
:author: Qizhi Li
"""
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
import time
import warnings
import argparse
import pandas as pd
from transformers import get_cosine_schedule_with_warmup

sys.path.append('..')
from static_data import file_path as fp
from models import BEMOJI


def load_chinese_data(emoji_list):
    """
    :param emoji_list: list
            emojis that need to be considering
    :return texts: list
    :return emojis: list
    """
    texts, emojis = [], []
    with open(fp.pre_train_chinese_data, encoding='utf-8') as f:
        content = f.readlines()

    i = 1
    for line in content:
        flag = True
        split_line = line.rstrip('\n').split('\t')

        # split multiple emojis, and deduplicate them
        emojis_in_line = set(split_line[1].split(','))

        # If emoji not in emoji_list, delete it
        for emoji in emojis_in_line:
            if emoji not in emoji_list:
                flag = False
                break

        i += 1

        if not flag:
            continue

        emojis.append(emojis_in_line)
        texts.append(split_line[0])

    return texts, emojis


def load_github_data():
    """
    :return texts: list
    :return emojis: list
    """
    texts, emojis = [], []
    with open(fp.pre_train_english_data, encoding='utf-8') as f:
        content = f.readlines()

    for line in content:
        split_line = line.rstrip('\n').split('\t')
        texts.append(split_line[0])
        emojis.append(split_line[1])

    return texts, emojis


def reconstruct_data():
    """
    Reconstruct text with emoji, if one text with multiple emojis (text[emoji1][emoji2]),
    then convert it into one text with one emoji (text[emoji1], text[emoji2])
    :return text_with_single_emoji: list
            One text only contains one emoji
            [(text1, emoji11), (text1, emoji12), (text2, emoji21), ...]
    """
    texts, emojis = load_chinese_data(emoji_list)
    text_with_single_emoji = []
    for text, emoji in zip(texts, emojis):
        for e in emoji:
            text_with_single_emoji.append((text, e))

    return text_with_single_emoji


def get_inputs_and_labels(mode, prefix):
    """
    将文本改为 prompt
    :param mode: str
            pre_train_english: pre-training github data
            pre_train_chinese: pre-training weibo data
    :param prefix: str
    :return prompt_inputs: list
            input sequence with prompt template
    :return emojis: list
            emoji descriptions for prompt learning
    :return mlm_inputs: list
            input sequence for MLM
    """
    prompt_inputs, mlm_inputs, emojis = [], [], []
    if mode == 'pre_train_chinese':
        text_with_single_emoji = reconstruct_data()
        for i in text_with_single_emoji:
            prompt_inputs.append('%s%s' % (prefix, i[0]))
            emojis.append(emoji_defs[i[1]])
            mlm_inputs.append(i[0])
    elif mode == 'pre_train_english':
        texts_list, emojis_list = load_github_data()
        for i in range(len(texts_list)):
            prompt_inputs.append('%s%s' % (prefix, texts_list[i]))
            emojis.append(emoji_defs[emojis_list[i]])

        mlm_inputs = texts_list

    return prompt_inputs, emojis, mlm_inputs


def get_data_iter(mode, batch_size):
    """
    package batch
    :param mode: str
            pre_train_english: fine tune github data
            pre_train_chinese: fine tune weibo data
    :param batch_size: int
    :return prompt_inputs_batches: list
    :return emojis_batches: list
    :return mlm_inputs_batches: list
    :return batch_count: int
    """
    prompt_inputs, emojis, mlm_inputs = get_inputs_and_labels(mode, prefix)
    batch_count = int(len(prompt_inputs) / batch_size)
    prompt_inputs_batches, emojis_batches, mlm_inputs_batches = [], [], []

    for i in range(batch_count):
        prompt_inputs_batches.append(prompt_inputs[i * batch_size: (i + 1) * batch_size])
        emojis_batches.append(emojis[i * batch_size: (i + 1) * batch_size])
        mlm_inputs_batches.append(mlm_inputs[i * batch_size: (i + 1) * batch_size])

    return prompt_inputs_batches, emojis_batches, mlm_inputs_batches, batch_count


def train(args, device):
    config = BEMOJI.FineTuneConfig()
    config.mlm_config()
    if args.mode == 'pre_train_english':
        config.pre_train_config(batch_size=150,
                                epochs=10,
                                learning_rate=1e-6,
                                weight_decay=1e-4,
                                from_path=config.bert_english_path,
                                device=device)
    elif args.mode == 'pre_train_chinese':
        config.pre_train_config(batch_size=128,
                                epochs=10,
                                learning_rate=1e-6,
                                weight_decay=1e-4,
                                from_path=config.bert_chinese_path,
                                device=device)

    prompt_inputs_batches, emojis_batches, mlm_inputs_batches, batch_count = get_data_iter(args.mode, config.batch_size)
    pretrain_model = BEMOJI.BEMOJI(args, config, device)

    optimizer = AdamW(pretrain_model.parameters(),
                      lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    schedule = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=len(prompt_inputs_batches),
                                               num_training_steps=config.epochs * len(prompt_inputs_batches))

    total_time = 0

    print('start pretraining')
    for epoch in range(config.epochs):
        index = 0
        loss_sum = 0
        pretrain_model.train()
        start = time.time()
        for i in range(batch_count):
            prompt_inputs = prompt_inputs_batches[i]
            emojis_inputs = emojis_batches[i]
            mlm_inputs = mlm_inputs_batches[i]

            loss = pretrain_model(mlm_inputs, emojis_inputs, prompt_inputs)
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

            index += 1
            if index % 100 == 0:
                print("Epoch {:02d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.0f}".format(
                    epoch + 1, index, batch_count, loss_sum / index, time.time() - start))

        end = time.time()
        print("epoch {:02d}, Loss {:.4f}, duration {:.4f}".format(
            epoch + 1, loss_sum / batch_count, end - start))
        total_time += end - start

        if args.mode == 'pre_train_chinese':
            torch.save(pretrain_model.context_bert.state_dict(),
                       os.path.join(fp.pre_train_parameters, 'BEMOJI_context_ep_{}.bin'.format(epoch + 1)))
            torch.save(pretrain_model.emoji_bert.state_dict(),
                       os.path.join(fp.pre_train_parameters, 'BEMOJI_emoji_ep_{}.bin'.format(epoch + 1)))
        else:
            torch.save(pretrain_model.context_bert.state_dict(),
                       os.path.join(fp.pre_train_parameters, 'BEMOJI_github_context_ep_{}.bin'.format(epoch + 1)))
            torch.save(pretrain_model.emoji_bert.state_dict(),
                       os.path.join(fp.pre_train_parameters, 'BEMOJI_github_emoji_ep_{}.bin'.format(epoch + 1)))

    print('total pretraining time: ', total_time)


warnings.filterwarnings('ignore')
prefix = '[MASK]'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--mode',
                    help='pre_train_english: fine tune English dataset,'
                         'pre_train_chinese: fine tune Chinese dataset',
                    default='pre_train_chinese')
parser.add_argument('--mean_sort', type=str, help='full or decay', default='full')
parser.add_argument('--mean_decay_param', type=float,
                    help='What decay to use with mean decay', default=1.0)
args = parser.parse_args()

if args.mode == 'pre_train_chinese':
    emoji2def = pd.read_csv(fp.weibo_emoji2def)
    emoji_list = list(emoji2def['微博表情'])
    emoji_defs = {}
    for _, line in emoji2def.iterrows():
        emoji_defs[line['微博表情']] = line['表情定义']
elif args.mode == 'pre_train_english':
    emoji2def = pd.read_csv(fp.github_emoji2def)
    emoji_defs = {}
    for _, line in emoji2def.iterrows():
        emoji_defs[line['emoji']] = line['definition']
else:
    raise ValueError('args.mode must be \'pre_train_english\' or \'pre_train_chinese\'')

train(args, device)
