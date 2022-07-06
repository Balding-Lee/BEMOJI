"""
Transformers for downstream tasks without emojis
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
from models import bert_non_emoji


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


def evaluate(model, batch_count, batch_X, batch_y, device, args):
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

            outputs = model(input_seqs)

            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    if args.dataset == 'SST-2':
        acc = accuracy_score(labels_all, predict_all)
        return loss_total / batch_count, acc
    elif args.dataset == 'ECISA':
        macro_P = precision_score(labels_all, predict_all, average='macro')
        macro_R = recall_score(labels_all, predict_all, average='macro')
        macro_F1 = f1_score(labels_all, predict_all, average='macro')
        return loss_total / batch_count, macro_P, macro_R, macro_F1
    else:
        acc = accuracy_score(labels_all, predict_all)
        macro_P = precision_score(labels_all, predict_all, average='macro')
        macro_R = recall_score(labels_all, predict_all, average='macro')
        macro_F1 = f1_score(labels_all, predict_all, average='macro')
        return loss_total / batch_count, acc, macro_P, macro_R, macro_F1


def train(args, device):
    config = bert_non_emoji.Config(
        args,
        output_size=dataset_num_labels[args.dataset]
    )

    save_path = os.path.join(fp.model_parameters, 'BERT_{}_parameters.bin'.format(args.dataset))
    datasets = bert_non_emoji.PreprocessDataset()
    if args.dataset == 'ECISA':
        early_stop = 512
        train_texts, train_labels = datasets.ECISA('train')
        dev_texts, dev_labels = datasets.ECISA('dev')
        test_texts, test_ids = datasets.ECISA('test')

        train_batch_X, train_batch_y, train_batch_count = get_data_iter(train_texts, train_labels, config.batch_size)
        dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(dev_texts, dev_labels, config.batch_size)
    elif args.dataset == 'SemEval':
        early_stop = 512
        train_texts, train_labels = datasets.SemEval('train')
        dev_texts, dev_labels = datasets.SemEval('dev')
        test_texts, test_labels = datasets.SemEval('test')

        train_batch_X, train_batch_y, train_batch_count = get_data_iter(train_texts, train_labels, config.batch_size)
        dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(dev_texts, dev_labels, config.batch_size)
        test_batch_X, test_batch_y, test_batch_count = get_data_iter(test_texts, test_labels, config.batch_size)
    elif args.dataset == 'NLPCC':
        early_stop = 512
        train_texts, train_labels = datasets.NLPCC('train')
        dev_texts, dev_labels = datasets.NLPCC('dev')
        test_texts, test_labels = datasets.NLPCC('test')

        train_batch_X, train_batch_y, train_batch_count = get_data_iter(train_texts, train_labels, config.batch_size)
        dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(dev_texts, dev_labels, config.batch_size)
        test_batch_X, test_batch_y, test_batch_count = get_data_iter(test_texts, test_labels, config.batch_size)

    model = bert_non_emoji.BERTCLS(args, config, device).to(device)
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

            y_hat = model(input_seqs)

            optimizer.zero_grad()
            l = loss(y_hat, labels)
            l.backward()
            optimizer.step()
            schedule.step()

            if (index + 1) % 100 == 0:
                if args.dataset == 'SST-2':
                    dev_loss, dev_acc = evaluate(model,
                                                 dev_batch_count,
                                                 dev_batch_X,
                                                 dev_batch_y,
                                                 device,
                                                 args)
                    print('Epoch %d | Iter %d |  Dev loss %f | Dev acc %f | '
                          'Duration %.2f' % (
                              epoch + 1, index + 1, dev_loss, dev_acc, time.time() - iter_start
                          ))
                elif args.dataset == 'ECISA':
                    dev_loss, dev_macro_P, dev_macro_R, dev_macro_F1 = evaluate(model,
                                                                                dev_batch_count,
                                                                                dev_batch_X,
                                                                                dev_batch_y,
                                                                                device,
                                                                                args)
                    print('Epoch %d | Iter %d |  Dev loss %f | Dev macro P %f | '
                          'Dev macro R %f | Dev macro F1 %f | Duration %.2f' % (
                              epoch + 1, index + 1, dev_loss, dev_macro_P,
                              dev_macro_R, dev_macro_F1, time.time() - iter_start
                          ))
                else:
                    dev_loss, dev_acc, dev_macro_P, dev_macro_R, dev_macro_F1 = evaluate(model,
                                                                                         dev_batch_count,
                                                                                         dev_batch_X,
                                                                                         dev_batch_y,
                                                                                         device,
                                                                                         args)
                    print('Epoch %d | Iter %d |  Dev loss %f | Dev acc %f |'
                          ' Dev macro P %f | Dev macro R %f | Dev macro F1 %f | Duration %.2f' % (
                              epoch + 1, index + 1, dev_loss, dev_acc, dev_macro_P,
                              dev_macro_R, dev_macro_F1, time.time() - iter_start
                          ))
                model.train()

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
    model = bert_non_emoji.BERTCLS(args, config, device).to(device)
    model.load_state_dict(torch.load(save_path))

    if args.dataset != 'SST-2' and args.dataset != 'ECISA':
        test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1 = evaluate(model,
                                                                                  test_batch_count,
                                                                                  test_batch_X,
                                                                                  test_batch_y,
                                                                                  device,
                                                                                  args)
        print('Test loss %f | Test acc %f | Test macro P %f | Test macro R %f | Test macro F1 %f' % (
            test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1
        ))
    elif args.dataset == 'ECISA':
        model.eval()
        preds = []
        for i in test_texts:
            output = model([i])
            preds.append(output.argmax().item())

        with open(os.path.join(fp.chinese_sa_dataset, 'SMP2019-ECISA/bert_result.txt'), 'a', encoding='utf-8') as f:
            for i in range(len(test_ids)):
                if i == len(test_ids) - 1:
                    f.write('%s\t%s' % (test_ids[i], preds[i]))
                else:
                    f.write('%s\t%s\n' % (test_ids[i], preds[i]))


warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--dataset',
                    help='SST-2, SemEval, NLPCC, ECISA',
                    default='ECISA')
args = parser.parse_args()

dataset_num_labels = {
    'SST-2': 2,
    'ECISA': 3,
    'NLPCC': 2,
    'SemEval': 3,
}

train(args, device)
