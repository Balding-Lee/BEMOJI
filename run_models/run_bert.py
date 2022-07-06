"""
Transformers for fine-tuning
:author: Qizhi Li
"""
import os
import sys
import warnings
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import get_cosine_schedule_with_warmup

sys.path.append('..')
from models import bert
from static_data import file_path as fp


def load_data(file_path):
    """
    :param file_path: str
    :return x: list
    :return y: list
    """
    csv_data = pd.read_csv(file_path)

    x = csv_data['data'].tolist()
    y = csv_data['label'].tolist()

    return x, y


def get_data_iter(x, y, batch_size):
    """
    Package batch
    :param x: list
    :param y: list
    :param batch_size: int
    :return batch_X: list
    :return batch_y: list
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


def evaluate(model, batch_count, batch_X, batch_y, device):
    # def evaluate(model, batch_count, batch_X, batch_y, batch_pos, implicit_lexicon, device):
    """
    Evaluating model on dev and test, and outputting loss, accuracy and macro-F1
    :param model: Object
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

            # outputs = model(texts, defs_ids, defs_indices)
            outputs = model(input_seqs)
            # outputs = model(inputs, batch_pos[i], implicit_lexicon)
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
        from_path = fp.weibo_data
        early_stop = 1024
        config = bert.Config(args, 2)

    else:
        from_path = fp.github_data
        early_stop = 1024
        config = bert.Config(args, 3, max_seq_length=128)

    train_x, train_y = load_data(
        os.path.join(from_path, 'train.csv'))
    dev_x, dev_y = load_data(
        os.path.join(from_path, 'dev.csv'))
    test_x, test_y = load_data(
        os.path.join(from_path, 'test.csv'))

    train_batch_X, train_batch_y, train_batch_count = get_data_iter(train_x,
                                                                    train_y,
                                                                    config.batch_size)
    dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(dev_x,
                                                              dev_y,
                                                              config.batch_size)
    test_batch_X, test_batch_y, test_batch_count = get_data_iter(test_x,
                                                                 test_y,
                                                                 config.batch_size)

    model = bert.BERT(args, config, device=device).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=len(train_batch_X),
                                               num_training_steps=config.epochs * len(train_batch_X))

    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    n = 0
    start = time.time()

    print('start training')
    for epoch in range(config.epochs):
        model.train()
        iter_start = time.time()
        for i in range(train_batch_count):
            inputs = train_batch_X[i]
            labels = torch.tensor(train_batch_y[i]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            schedule.step()

            if (n + 1) % 100 == 0:
                dev_loss, dev_acc, dev_macro_P, dev_macro_R, dev_macro_f1 = evaluate(model,
                                                                                     dev_batch_count,
                                                                                     dev_batch_X,
                                                                                     dev_batch_y,
                                                                                     device)
                model.train()
                print('Epoch %d | Iter %d |  Dev loss %f | Dev acc %f | Dev macro P %f | '
                      'Dev macro R %f | Dev macro F1 %f | Duration %.2f' % (
                          epoch + 1, n + 1, dev_loss, dev_acc, dev_macro_P,
                          dev_macro_R, dev_macro_f1, time.time() - iter_start
                      ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = n

            if n - last_improve > early_stop:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds 256 batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            n += 1
        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))

    model = bert.BERT(args, config, device=device).to(device)
    model.load_state_dict(torch.load(config.save_path))
    test_loss, test_acc, test_macro_P, test_macro_R, test_macro_f1 = evaluate(model,
                                                                              test_batch_count,
                                                                              test_batch_X,
                                                                              test_batch_y,
                                                                              device)
    print('Test loss %f | Test acc %f | Test macro P %f | Test macro R %f | Test macro F1 %f' % (
        test_loss, test_acc, test_macro_P, test_macro_R, test_macro_f1
    ))


warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model',
                    help='BERT_base, BERT_large, RoBERTa_base, RoBERTa_large, '
                         'DistilBert_base, XLNet_base, XLNet_mid, XLNet_large. '
                         'Note: only english dataset has BERT_large, XLNet_large, '
                         'and only chinese dataset has XLNet_mid')
parser.add_argument('-d',
                    '--dataset',
                    help='chinese: Chinese dataset, JavaLib: English dataset, '
                         'Jira: English dataset, CodeReview: English dataset, '
                         'StackOverflow: English dataset, Github: Github dataset',
                    default='chinese')
parser.add_argument('-p',
                    '--pretrain_epoch',
                    help='0: w/o fine tune, 1-10: fine tune parameters',
                    type=int,
                    default=0)
parser.add_argument('-f',
                    '--fine_tune',
                    type=int,
                    help='0: only fine tune classifier, 1: fine tune all parameters',
                    default=1)
args = parser.parse_args()
train(args, device)
