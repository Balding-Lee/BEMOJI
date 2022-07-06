"""
BEMOJI for fine-tuning
:author: Qizhi Li
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import argparse
import warnings
from transformers import get_cosine_schedule_with_warmup

sys.path.append('..')
from static_data import file_path as fp
from models import BEMOJI_CLS


def load_data(file_path):
    """
    load data
    :param file_path: str
    :return x: list
            input texts
    :return emoji: list
            input emojis
    :return y: list
            true labels
    """
    csv_data = pd.read_csv(file_path)

    x = csv_data['data'].tolist()
    emoji = csv_data['emoji'].tolist()
    y = csv_data['label'].tolist()

    return x, emoji, y


def load_english_data(file_path):
    """
    load English data
    :param file_path: str
    :return x: list
            input texts
    :return emoji: list
            input emojis
    :return y: list
            true labels
    """
    csv_data = pd.read_csv(file_path)

    x = csv_data['data'].tolist()
    y = csv_data['label'].tolist()

    return x, y


def get_data_iter(x, emoji, y, batch_size):
    """
    package batch
    :param x: list
    :param emoji: list
    :param y: list
    :param batch_size: int
    :return batch_X: list
    :return batch_y: list
    :return batch_emoji: list
    :return batch_count: int
    """
    batch_count = int(len(x) / batch_size)
    batch_X, batch_y, batch_emoji = [], [], []

    for i in range(batch_count):
        batch_X.append(x[i * batch_size: (i + 1) * batch_size])
        batch_y.append(y[i * batch_size: (i + 1) * batch_size])
        batch_emoji.append(emoji[i * batch_size: (i + 1) * batch_size])

    return batch_X, batch_y, batch_emoji, batch_count


def get_english_data_iter(x, y, batch_size):
    """
    package english batch
    :param x: list
    :param y: list
    :param batch_size: int
    :return batch_X: list
    :return batch_y: list
    :return batch_emoji: list
    :return batch_count: int
    """
    if len(x) % batch_size != 0:
        flag = False
        batch_count = int(len(x) / batch_size) + 1
    else:
        flag = True
        batch_count = int(len(x) / batch_size)

    batch_X, batch_y = [], []

    if flag:
        for i in range(batch_count):
            batch_X.append(x[i * batch_size: (i + 1) * batch_size])
            batch_y.append(y[i * batch_size: (i + 1) * batch_size])
    else:
        for i in range(batch_count):
            if i == batch_count - 1:
                batch_X.append(x[i * batch_size:])
                batch_y.append(y[i * batch_size:])
            else:
                batch_X.append(x[i * batch_size: (i + 1) * batch_size])
                batch_y.append(y[i * batch_size: (i + 1) * batch_size])

    return batch_X, batch_y, batch_count


def get_emoji_definitions(dataset, emojis, emoji2def):
    """
    covert emoji to emoji description
    :param dataset: str
    :param emojis: list
            ['emoji11', 'emoji21, emoji22', ...]
    :param emoji2def: dict
    :return emoji_defs: list
    """
    if dataset == 'chinese':
        emoji_defs = []
        for emoji in emojis:
            emoji_list = emoji.split(',')
            emoji_def = []
            for e in emoji_list:
                emoji_def.append(emoji2def[e])
            emoji_defs.append(emoji_def)
    elif dataset == 'Github':
        emoji_defs = []
        for emoji in emojis:
            emoji_defs.append(emoji2def[emoji])

    return emoji_defs


def evaluate(model, batch_count, batch_X, batch_y, device, args, batch_emoji=None, emoji2def=None):
    # def evaluate(model, batch_count, batch_X, batch_y, batch_pos, implicit_lexicon, device):
    """
    Evaluating model on dev and test, and outputting loss, accuracy and macro-F1
    :param model: Object
    :param batch_count: int
    :param batch_X: list
    :param batch_y: list
    :param device: 'cuda' or 'cpu'
    :param args: Object
    :param batch_emoji: list or None
            default: None
    :param emoji2def: dict or None
            default: None
    :return: float
            total loss
    :return macro_P: float
            total macro_P
    :return macro_R: float
            total macro_R
    :return macro_F1: float
            total macro-F1
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for i in range(batch_count):
            input_seqs = batch_X[i]
            labels = torch.tensor(batch_y[i]).to(device)

            emojis = batch_emoji[i]
            input_emojis = get_emoji_definitions(args.dataset, emojis, emoji2def)
            outputs = model(input_seqs, input_emojis)

            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    accuracy = accuracy_score(labels_all, predict_all)
    macro_P = precision_score(labels_all, predict_all, average='macro')
    macro_R = recall_score(labels_all, predict_all, average='macro')
    macro_F1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / batch_count, accuracy, macro_P, macro_R, macro_F1


def train(args, device):

    if args.dataset == 'chinese':
        early_stop = 512
        save_path = os.path.join(fp.model_parameters, 'BEMOJI_base_parameters.bin')
        from_path = fp.weibo_data

        emoji2def_csv = pd.read_csv(fp.weibo_emoji2def)
        emoji2def = {}
        for _, line in emoji2def_csv.iterrows():
            emoji2def[line['微博表情']] = line['表情定义']

        config = BEMOJI_CLS.Config(
            args,
            output_size=2,
            pre_train_epoch=args.pre_train_epoch,
            parameter_path=fp.pre_train_parameters
        )

    else:
        early_stop = 1024
        save_path = os.path.join(fp.model_parameters,
                                 'BEMOJI_{}_base_parameters.bin'.format(args.dataset))
        from_path = fp.github_data

        emoji2def_csv = pd.read_csv(fp.github_emoji2def)
        emoji2def = {}
        for _, line in emoji2def_csv.iterrows():
            emoji2def[line['emoji']] = line['definition']

        config = BEMOJI_CLS.Config(
            args,
            output_size=2,
            pre_train_epoch=args.pre_train_epoch,
            parameter_path=fp.pre_train_parameters,
            texts_max_length=128,
            batch_size=32
        )

    train_x, train_emoji, train_y = load_data(
        os.path.join(from_path, 'train.csv'))
    dev_x, dev_emoji, dev_y = load_data(
        os.path.join(from_path, 'dev.csv'))
    test_x, test_emoji, test_y = load_data(
        os.path.join(from_path, 'test.csv'))

    train_batch_X, train_batch_y, train_batch_emoji, train_batch_count = get_data_iter(train_x,
                                                                                       train_emoji,
                                                                                       train_y,
                                                                                       config.batch_size)
    dev_batch_X, dev_batch_y, dev_batch_emoji, dev_batch_count = get_data_iter(dev_x,
                                                                               dev_emoji,
                                                                               dev_y,
                                                                               config.batch_size)
    test_batch_X, test_batch_y, test_batch_emoji, test_batch_count = get_data_iter(test_x,
                                                                                   test_emoji,
                                                                                   test_y,
                                                                                   config.batch_size)

    model = BEMOJI_CLS.BEMOJICLS(args, config, device).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=len(train_batch_X),
                                               num_training_steps=config.epochs * len(train_batch_X))

    require_improvement = early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    start = time.time()

    index = 0
    for epoch in range(config.epochs):
        model.train()
        iter_start = time.time()
        for i in range(train_batch_count):
            input_seqs = train_batch_X[i]
            labels = torch.tensor(train_batch_y[i]).to(device)

            emojis = train_batch_emoji[i]
            input_emojis = get_emoji_definitions(args.dataset, emojis, emoji2def)
            y_hat = model(input_seqs, input_emojis)

            optimizer.zero_grad()
            l = loss(y_hat, labels)
            l.backward()
            optimizer.step()
            schedule.step()

            if (index + 1) % 100 == 0:

                dev_loss, dev_acc, dev_macro_P, dev_macro_R, dev_macro_F1 = evaluate(model,
                                                                                     dev_batch_count,
                                                                                     dev_batch_X,
                                                                                     dev_batch_y,
                                                                                     device,
                                                                                     args,
                                                                                     dev_batch_emoji,
                                                                                     emoji2def)
                model.train()
                print('Epoch %d | Iter %d |  Dev loss %f | Dev acc %f | Dev macro P %f | '
                      'Dev macro R %f | Dev macro F1 %f | Duration %.2f' % (
                          epoch + 1, index + 1, dev_loss, dev_acc, dev_macro_P,
                          dev_macro_R, dev_macro_F1, time.time() - iter_start
                      ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    last_improve = index

            if index - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds args.early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            index += 1

        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))
    model = BEMOJI_CLS.BEMOJICLS(args, config, device).to(device)
    model.load_state_dict(torch.load(save_path))

    test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1 = evaluate(model,
                                                                              test_batch_count,
                                                                              test_batch_X,
                                                                              test_batch_y,
                                                                              device,
                                                                              args,
                                                                              test_batch_emoji,
                                                                              emoji2def)

    print('Test loss %f | Test acc %f | Test macro P %f | Test macro R %f | Test macro F1 %f' % (
        test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1
    ))


warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

parser.add_argument('-d',
                    '--dataset',
                    help='chinese: Chinese dataset, Github: Github dataset',
                    default='chinese')
parser.add_argument('-p',
                    '--pre_train_epoch',
                    type=int,
                    help='0-10, 0: w/o fine tune',
                    default=0)
parser.add_argument('-f',
                    '--fine_tune',
                    type=int,
                    help='0: only fine tune classifier, 1: fine tune all parameters',
                    default=1)
parser.add_argument('-a',
                    '--ablation',
                    help='None: without ablation, we: without emoji, wf: without fusion layer',
                    default='None')
args = parser.parse_args()
train(args, device)
