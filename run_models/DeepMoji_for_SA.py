"""
DeepMoji for downstream tasks
:author: Qizhi Li
"""
import argparse
import os
import sys
import time
import pyprind
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import warnings

sys.path.append('..')
import utils
from static_data import file_path as fp
from models import DeepMoji


def load_data(file_path):
    """
    load data
    :param file_path: str
    :return x: list
    :return y: list
    """
    csv_data = pd.read_csv(file_path)

    x = csv_data['data'].tolist()
    y = csv_data['label'].tolist()

    return x, y


def get_wordset_and_lables(texts):
    """
    obtaining word set
    :param texts: list
    :return word_set: set
    """
    word_set = set()
    for text in texts:
        word_set = word_set.union(list(text))

    return word_set


def get_all_words_embed(w2v_embed, words_set):
    """
    obtaining word embedding
    :param w2v_embed: Object
    :param words_set: set
    :return embed: FloatTensor
    """
    embed = torch.FloatTensor(len(words_set), 300)
    i = 0

    pper = pyprind.ProgPercent(len(words_set))
    for word in words_set:
        try:
            embed[i] = torch.from_numpy(w2v_embed[word].copy())
        except KeyError:
            embed[i] = torch.from_numpy(w2v_embed['UNK'].copy())

        i += 1
        pper.update()

    # '<PAD>' embedding is 0
    embed = torch.cat((embed, torch.zeros(1, embed.shape[1])))

    return embed


def change_sentence_to_ids(args, dataset, word2id):
    """
    convert words in sentence to id
    :param args: Object
    :param dataset: list
    :param word2id: dict
            {word: id}
    :return: sentences_ids: list
    """
    if args.mode == 'Weibo':
        sentences_ids = []
        for sentence in dataset:
            words = list(sentence)
            sentence_ids = []
            for word in words:
                sentence_ids.append(word2id[word])
            sentences_ids.append(sentence_ids)
    else:
        sentences_ids = []
        for sentence in dataset:
            words = sentence.split(' ')
            sentence_ids = []
            for word in words:
                sentence_ids.append(word2id[word])
            sentences_ids.append(sentence_ids)

    return sentences_ids


def padding_or_truncate(sentences_ids, padding, max_seq_length):
    """
    :param sentences_ids: list
    :param padding: int
            '<PAD>' 的 id
    :param max_seq_length: int
    :return X: list
    """
    X = []

    for sentence in sentences_ids:
        if len(sentence) > max_seq_length:
            X.append(sentence[: max_seq_length])
        else:
            pt = sentence.copy()
            pt.extend([padding] * (max_seq_length - len(sentence)))
            X.append(pt)

    return X


def get_data_iter(args, X, y, word2id, padding, config, data_mode):
    """
    package batch
    :param args: Object
    :param X: list
    :param y: list
    :param word2id: dict
    :param padding: int
    :param config: Object
    :param data_mode: str
            'train': DataLoader shuffle=True
            'dev' and 'test': DataLoader shuffle=False
    :return data_iter: DataLoader
    """
    X_ids = change_sentence_to_ids(args, X, word2id)

    X_ids = padding_or_truncate(X_ids, padding, config.max_seq_length)

    X_ids = torch.LongTensor(X_ids)
    y_ids = torch.LongTensor(y)

    dataset = Data.TensorDataset(X_ids, y_ids)

    if data_mode == 'train':
        data_iter = Data.DataLoader(dataset, config.batch_size, shuffle=True)
    else:
        data_iter = Data.DataLoader(dataset, config.batch_size, shuffle=False)

    return data_iter


def evaluate(model, data_iter, device):
    """
    Evaluating model on dev and test, and outputting loss, accuracy and macro-F1
    :param model: Object
    :param data_iter: DataLoader
            dev or test
    :param device: Object
    :return: float
            total loss
    :return acc: float
            total accuracy
    :return macro_P: float
            total macro P
    :return macro_R: float
            total macro R
    :return macro_F1: float
            total macro F1
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            texts = texts.to(device)
            labels = labels.to(device)

            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = accuracy_score(labels_all, predict_all)
    macro_P = precision_score(labels_all, predict_all, average='macro')
    macro_R = recall_score(labels_all, predict_all, average='macro')
    macro_F1 = f1_score(labels_all, predict_all, average='macro')
    return loss_total / len(data_iter), acc, macro_P, macro_R, macro_F1


def train(device):
    save_path = os.path.join(fp.model_parameters, 'deepmoji.bin')

    if args.mode == 'Weibo':
        train_x, train_y = load_data(
            os.path.join(fp.weibo_data, 'train.csv'))
        dev_x, dev_y = load_data(
            os.path.join(fp.weibo_data, 'dev.csv'))
        test_x, test_y = load_data(
            os.path.join(fp.weibo_data, 'test.csv'))
        w2v_embed = utils.read_file('pkl', fp.chinese_word2vec)
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'weibo_word2id.json'))

        texts = []
        texts.extend(train_x)
        texts.extend(dev_x)
        texts.extend(test_x)

        words_set = get_wordset_and_lables(texts)
        embed = get_all_words_embed(w2v_embed, words_set)
    else:
        train_x, train_y = load_data(
            os.path.join(fp.github_data, 'train.csv'))
        dev_x, dev_y = load_data(
            os.path.join(fp.github_data, 'dev.csv'))
        test_x, test_y = load_data(
            os.path.join(fp.github_data, 'test.csv'))
        w2v_embed = KeyedVectors.load_word2vec_format(fp.english_word2vec, binary=True)
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'github_word2id.json'))

        words_set = list(word2id.keys())
        embed = get_all_words_embed(w2v_embed, words_set)

    padding = word2id['<PAD>']

    config = DeepMoji.Config(num_classes=2)
    config.lr = 1e-5
    config.batch_size = 64

    train_iter = get_data_iter(args, train_x, train_y, word2id, padding, config, 'train')
    dev_iter = get_data_iter(args, dev_x, dev_y, word2id, padding, config, 'dev')
    test_iter = get_data_iter(args, test_x, test_y, word2id, padding, config, 'test')
    
    model = DeepMoji.DeepMoji(embed, config, device).to(device)
    if args.mode == 'Weibo':
        model.lstm_0.load_state_dict(torch.load(os.path.join(fp.pre_train_parameters,
                                                             'deepmoji_lstm0.bin')),
                                     strict=False)
        model.lstm_1.load_state_dict(torch.load(os.path.join(fp.pre_train_parameters,
                                                             'deepmoji_lstm1.bin')),
                                     strict=False)
        model.attention.load_state_dict(torch.load(os.path.join(fp.pre_train_parameters,
                                                                'deepmoji_attention.bin')),
                                        strict=False)
    else:
        model.lstm_0.load_state_dict(torch.load(os.path.join(fp.pre_train_parameters,
                                                             'deepmoji_github_lstm0.bin')),
                                     strict=False)
        model.lstm_1.load_state_dict(torch.load(os.path.join(fp.pre_train_parameters,
                                                             'deepmoji_github_lstm1.bin')),
                                     strict=False)
        model.attention.load_state_dict(torch.load(os.path.join(fp.pre_train_parameters,
                                                                'deepmoji_github_attention.bin')),
                                        strict=False)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-6)

    require_improvement = config.early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    i = 0

    print('start training')
    start = time.time()
    for epoch in range(config.epoch):
        model.train()
        iter_start = time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)

            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                pred = torch.max(y_hat.data.cpu(), 1)[1].numpy()
                acc = accuracy_score(y.detach().cpu().numpy(), pred)

                dev_loss, dev_acc, dev_macro_P, dev_macro_R, dev_macro_F1 = evaluate(model, dev_iter, device)

                model.train()

                print('Epoch %d | Iter %d | Train loss %f | Dev loss %f | Dev acc %f | Dev macro P %f | '
                      'Dev macro R %f | Dev macro F1 %f | Duration %.2f' % (
                          epoch + 1, i + 1, l.item(), dev_loss, dev_acc, dev_macro_P,
                          dev_macro_R, dev_macro_F1, time.time() - iter_start
                      ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    last_improve = i
                model = model.to(device)

            if i - last_improve > require_improvement:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds args.early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            i += 1
        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))

    model = DeepMoji.DeepMoji(embed, config, device).to(device)
    model.load_state_dict(torch.load(save_path))

    test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1 = evaluate(model, test_iter, device)
    print('Test loss %f | Test acc %f | Test macro P %f | Test macro R %f | Test macro F1 %f' % (
        test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1
    ))


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--mode',
                    help='Github, Weibo',
                    default='Weibo')
args = parser.parse_args()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
train(device)
