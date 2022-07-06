"""
BEMOJI for glue
:author: Qizhi Li
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import argparse
import warnings
from transformers import get_cosine_schedule_with_warmup

sys.path.append('..')
from static_data import file_path as fp
from models import BEMOJI_GLUE


def get_data_iter(a, y, batch_size, b=None):
    """
    package batch
    :param a: list
    :param y: list
    :param batch_size: int
    :param b: list or None
            if has two sentence sequences, b is a list, else None
            default None
    :return batch_a: list
    :return batch_b: list
            if b is a list
    :return batch_y: list
    :return batch_count: int
    """
    batch_count = int(len(a) / batch_size)
    batch_a, batch_y = [], []

    for i in range(batch_count):
        batch_a.append(a[i * batch_size: (i + 1) * batch_size])
        batch_y.append(y[i * batch_size: (i + 1) * batch_size])

    if b is not None:
        batch_b = []
        for i in range(batch_count):
            batch_b.append(b[i * batch_size: (i + 1) * batch_size])

        return batch_a, batch_b, batch_y, batch_count
    else:
        return batch_a, batch_y, batch_count


def acc(pred, label):
    return accuracy_score(label, pred)


def f1(pred, label):
    return f1_score(label, pred, average='macro')


def MCC(pred, label):
    return matthews_corrcoef(label, pred)


def Pearsonr(pred, label):
    return pearsonr(pred, label)[0]


def evaluate(model, batch_count, batch_X, batch_y, device, args, batch_b=None):
    """
    Evaluating model on dev and test
    :param model: Object
    :param batch_count: int
    :param batch_X: list
    :param batch_y: list
    :param device: 'cuda' or 'cpu'
    :param args: Object
    :param batch_b: list or None
            if the dataset has two sentence sequences, batch_b is a list, else None
            default: None
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for i in range(batch_count):
            labels = torch.tensor(batch_y[i]).to(device)

            if batch_b:
                input_seqs1 = batch_X[i]
                input_seqs2 = batch_b[i]
                outputs = model(input_seqs1, input_seqs2)
            else:
                input_seqs = batch_X[i]
                outputs = model(input_seqs)

            if args.dataset == 'STS-B':
                loss = F.mse_loss(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)

            loss_total += loss
            labels = labels.data.cpu().numpy()
            if args.dataset != 'STS-B':
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            else:
                predic = outputs.data.cpu().numpy()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    if args.dataset == 'RTE' or args.dataset == 'SST-2' or args.dataset == 'QNLI' or \
            args.dataset == 'MNLI-matched' or args.dataset == 'MNLI-mismatched' or args.dataset == 'WNLI':
        return loss_total / batch_count, acc(predict_all, labels_all)
    elif args.dataset == 'QQP' or args.dataset == 'MRPC':
        return loss_total / batch_count, acc(predict_all, labels_all), f1(predict_all, labels_all)
    elif args.dataset == 'CoLA' or args.dataset == 'AX':
        return loss_total / batch_count, MCC(predict_all, labels_all)
    elif args.dataset == 'STS-B':
        return loss_total / batch_count, Pearsonr(predict_all, labels_all)
    else:
        return None


def train(args, device):

    config = BEMOJI_GLUE.Config(
        output_size=glue_tasks_num_labels[args.dataset],
        pre_train_epoch=args.pre_train_epoch,
        parameter_path=fp.pre_train_parameters,
        texts_max_length=128,
        batch_size=128
    )

    glue_dataset = BEMOJI_GLUE.PreprocessGlue()
    train_batch_a, train_batch_X = None, None

    if args.dataset == 'CoLA':
        train_x, train_y = glue_dataset.CoLA('train')
        dev_x, dev_y = glue_dataset.CoLA('dev')
        test_x, test_index = glue_dataset.CoLA('test')

        train_batch_X, train_batch_y, train_batch_count = get_data_iter(train_x,
                                                                        train_y,
                                                                        config.batch_size)
        dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(dev_x,
                                                                  dev_y,
                                                                  config.batch_size)
    elif args.dataset == 'MNLI-matched':
        train_premise, train_hypothesis, train_label = glue_dataset.MNLI_m('train')
        dev_matched_premise, dev_matched_hypothesis, dev_matched_label = glue_dataset.MNLI_m('dev_matched')

        test_text_a, test_text_b, test_index = glue_dataset.MNLI_m('test_matched')

        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_premise,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_hypothesis)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_matched_premise,
                                                                               dev_matched_label,
                                                                               config.batch_size,
                                                                               b=dev_matched_hypothesis)
        config.lr = 1e-5
    elif args.dataset == 'MNLI-mismatched':
        train_premise, train_hypothesis, train_label = glue_dataset.MNLI_mm('train')
        dev_mismatched_premise, dev_mismatched_hypothesis, dev_mismatched_label = glue_dataset.MNLI_mm('dev_mismatched')
        test_text_a, test_text_b, test_index = glue_dataset.MNLI_mm('test_mismatched')
        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_premise,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_hypothesis)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_mismatched_premise,
                                                                               dev_mismatched_label,
                                                                               config.batch_size,
                                                                               b=dev_mismatched_hypothesis)
        config.lr = 1e-5
    elif args.dataset == 'MRPC':
        train_text_a, train_text_b, train_label = glue_dataset.MRPC('train')
        dev_text_a, dev_text_b, dev_label = glue_dataset.MRPC('dev')
        test_text_a, test_text_b, test_label = glue_dataset.MRPC('test')

        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_label,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
    elif args.dataset == 'QNLI':
        train_text_a, train_text_b, train_label = glue_dataset.QNLI('train')
        dev_text_a, dev_text_b, dev_label = glue_dataset.QNLI('dev')
        test_text_a, test_text_b, test_index = glue_dataset.QNLI('test')

        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_label,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
        config.lr = 1e-5
        config.batch_size = 256
    elif args.dataset == 'QQP':
        config.lr = 1e-5
        train_text_a, train_text_b, train_label = glue_dataset.QQP('train')
        dev_text_a, dev_text_b, dev_label = glue_dataset.QQP('dev')
        test_text_a, test_text_b, test_index = glue_dataset.QQP('test')

        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_label,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
    elif args.dataset == 'RTE':
        train_text_a, train_text_b, train_label = glue_dataset.RTE('train')
        dev_text_a, dev_text_b, dev_label = glue_dataset.RTE('dev')
        test_text_a, test_text_b, test_index = glue_dataset.RTE('test')

        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_label,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
    elif args.dataset == 'SST-2':
        train_text, train_label = glue_dataset.SST2('train')
        dev_text, dev_label = glue_dataset.SST2('dev')
        test_x, test_index = glue_dataset.SST2('test')

        train_batch_X, train_batch_y, train_batch_count = get_data_iter(train_text,
                                                                                       train_label,
                                                                                       config.batch_size)
        dev_batch_X, dev_batch_y, dev_batch_count = get_data_iter(dev_text,
                                                                               dev_label,
                                                                               config.batch_size)
    elif args.dataset == 'STS-B':
        config.lr = 1e-4
        train_text_a, train_text_b, train_score = glue_dataset.STSB('train')
        dev_text_a, dev_text_b, dev_score = glue_dataset.STSB('dev')
        test_text_a, test_text_b, test_index = glue_dataset.STSB('test')

        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_score,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_score,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
    elif args.dataset == 'WNLI':
        train_text_a, train_text_b, train_score = glue_dataset.WNLI('train')
        dev_text_a, dev_text_b, dev_score = glue_dataset.WNLI('dev')
        test_text_a, test_text_b, test_index = glue_dataset.WNLI('test')

        config.batch_size = 16
        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_score,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_score,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
    elif args.dataset == 'AX':
        train_text_a, train_text_b, train_label = glue_dataset.AX('train')
        dev_text_a, dev_text_b, dev_label = glue_dataset.AX('dev_matched')
        test_text_a, test_text_b, test_index = glue_dataset.AX('test')
        train_batch_a, train_batch_b, train_batch_y, train_batch_count = get_data_iter(train_text_a,
                                                                                       train_label,
                                                                                       config.batch_size,
                                                                                       b=train_text_b)
        dev_batch_a, dev_batch_b, dev_batch_y, dev_batch_count = get_data_iter(dev_text_a,
                                                                               dev_label,
                                                                               config.batch_size,
                                                                               b=dev_text_b)
    else:
        raise ValueError('dataset must be CoLA, MNLI-matched, MNLI-mismatched,'
                         ' MRPC, QNLI, QQP, RTE, SST-2, or STS-B')

    early_stop = 1024
    save_path = os.path.join(fp.model_parameters,
                             'BEMOJI_{}_base_parameters.bin'.format(args.dataset))

    model = BEMOJI_GLUE.BEMOJICLS(args, config, device).to(device)

    if args.dataset != 'STS-B':
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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

            labels = torch.tensor(train_batch_y[i]).to(device)

            if train_batch_a:
                input_seqs1 = train_batch_a[i]
                input_seqs2 = train_batch_b[i]
                y_hat = model(input_seqs1, input_seqs2)
            else:
                input_seqs = train_batch_X[i]
                y_hat = model(input_seqs)

            optimizer.zero_grad()
            l = loss(y_hat, labels)
            l.backward()
            optimizer.step()

            if (index + 1) % 100 == 0:

                if args.dataset == 'CoLA' or args.dataset == 'SST-2':
                    results = evaluate(model, dev_batch_count, dev_batch_X, dev_batch_y, device, args)
                    dev_loss = results[0]
                    if args.dataset == 'CoLA':
                        print('Iter %d | Dev loss %f | MCC %f' % (
                            index + 1, dev_loss, results[1]
                        ))
                    else:
                        print('Iter %d | Dev loss %f | ACC %f' % (
                            index + 1, dev_loss, results[1]
                        ))
                else:
                    results = evaluate(model, dev_batch_count, dev_batch_a,
                                       dev_batch_y, device, args, batch_b=dev_batch_b)
                    dev_loss = results[0]
                    if args.dataset == 'RTE' or args.dataset == 'QNLI' or args.dataset == 'WNLI'\
                            or args.dataset == 'MNLI-matched' or args.dataset == 'MNLI-mismatched':
                        print('Iter %d | Dev loss %f | ACC %f' % (
                            index + 1, dev_loss, results[1]
                        ))
                    elif args.dataset == 'QQP' or args.dataset == 'MRPC':
                        print('Iter %d | Dev loss %f | ACC %f | F1 %f' % (
                            index + 1, dev_loss, results[1], results[2]
                        ))
                    elif args.dataset == 'AX':
                        print('Iter %d | Dev loss %f | MCC %f' % (
                            index + 1, dev_loss, results[1]
                        ))
                    else:
                        print('Iter %d | Dev loss %f | Pearsonr % f' % (
                            index + 1, dev_loss, results[1]
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
    model = BEMOJI_GLUE.BEMOJICLS(args, config, device).to(device)
    model.load_state_dict(torch.load(save_path))

    model.eval()

    if args.dataset != 'STS-B':
        if args.dataset == 'CoLA' or args.dataset == 'SST-2':
            pred = []
            for x in test_x:
                output = model([x])
                pred.append(output.argmax().item())
        else:
            pred = []
            for i in range(len(test_text_a)):
                output = model([test_text_a[i]], [test_text_b[i]])
                if args.dataset == 'MNLI-matched' or args.dataset == 'MNLI-mismatched' or args.dataset == 'AX':
                    pred.append(glue_dataset.AX_MNLI_id2label[output.argmax().item()])
                elif args.dataset == 'RTE' or args.dataset == 'QNLI':
                    pred.append(glue_dataset.QNLI_RTE_id2label[output.argmax().item()])
                else:
                    pred.append(output.argmax().item())
    else:
        pred = []
        for i in range(len(test_text_a)):
            output = model([test_text_a[i]], [test_text_b[i]])
            pred.append(output)

    if args.dataset == 'CoLA':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/CoLA.tsv'), index=False, sep='\t')
    elif args.dataset == 'MNLI-matched':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/MNLI-m.tsv'), index=False, sep='\t')
    elif args.dataset == 'MNLI-mismatched':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/MNLI-mm.tsv'), index=False, sep='\t')
    elif args.dataset == 'MRPC':
        df = pd.DataFrame({'IDS': list(range(len(pred))), 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/MRPC.tsv'), index=False, sep='\t')
    elif args.dataset == 'QNLI':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/QNLI.tsv'), index=False, sep='\t')
    elif args.dataset == 'QQP':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/QQP.tsv'), index=False, sep='\t')
    elif args.dataset == 'RTE':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/RTE.tsv'), index=False, sep='\t')
    elif args.dataset == 'SST-2':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/SST-2.tsv'), index=False, sep='\t')
    elif args.dataset == 'STS-B':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/STS-B.tsv'), index=False, sep='\t')
    elif args.dataset == 'WNLI':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/WNLI.tsv'), index=False, sep='\t')
    elif args.dataset == 'AX':
        df = pd.DataFrame({'IDs': test_index, 'labels': pred})
        df.to_csv(os.path.join(fp.glue_data, 'glue_results/AX.tsv'), index=False, sep='\t')

warnings.filterwarnings('ignore')
glue_tasks_num_labels = {
    "CoLA": 2,
    "MNLI-matched": 3,
    "MNLI-mismatched": 3,
    "MRPC": 2,
    "SST-2": 2,
    "STS-B": 1,
    "QQP": 2,
    "QNLI": 2,
    "RTE": 2,
    'WNLI': 2,
    'AX': 3,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-d',
                    '--dataset',
                    help='CoLA, MNLI-matched, MNLI-mismatched, MRPC, QNLI, QQP, RTE, SST-2, STS-B, WNLI')
parser.add_argument('-p',
                    '--pre_train_epoch',
                    type=int,
                    help='0-10, 0: w/o fine tune',
                    default=0)
args = parser.parse_args()
train(args, device)
