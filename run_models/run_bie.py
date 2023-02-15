"""
DeepMoji for sentiment analysis
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
from models import WATT_BiE_LSTM, MATT_BiE_LSTM


def load_data(file_path):
    """
    :param file_path: str
    :return x: list
            texts
    :return y: list
            true labels
    """
    csv_data = pd.read_csv(file_path)

    x = csv_data['data'].tolist()
    e = csv_data['emoji'].tolist()
    y = csv_data['label'].tolist()

    return x, e, y


def get_wordset_and_lables(texts):
    """
    :param texts: list
    :return word_set: set
    """
    word_set = set()
    for text in texts:
        word_set = word_set.union(list(text))

    return word_set


def get_all_words_embed(w2v_embed, words_set):
    """
    Obtain the word embedding of embedding layer
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

    # '<PAD>' is a zero vector
    embed = torch.cat((embed, torch.zeros(1, embed.shape[1])))

    return embed


def change_sentence_to_ids(args, dataset, word2id):
    """
    Convert each word in sentence to id
    :param dataset: list
    :param word2id: dict
            {word: id}
    :return sentences_ids: list
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


def change_emoji_to_ids(e, emoji2id):
    """
    Convert emojis in sentence to id
    :param e: list
    :param emoji2id: dict
        {emoji: id}
    :return emoji_ids: list
            [[emoji_id11, emoji_id12], [emoji_id21], ...]
    """
    emoji_ids = []
    for emojis in e:
        emojis_list = emojis.split(',')
        emoji_id = []
        for emoji in emojis_list:
            emoji_id.append(emoji2id[emoji])

        emoji_ids.append(emoji_id)

    return emoji_ids


def padding_or_truncate(sentences_ids, padding, max_seq_length):
    """
    :param sentences_ids: list
    :param padding: int
            The id of '<PAD>'
    :param max_seq_length: int
    :return:
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


def get_data_iter(args, X, e, y, word2id, emoji2id, padding, config):
    """
    Obtain DataLoader
    :param X: list
    :param y: list
    :param word2id: dict
    :param padding: int
    :param config: Object
    :return data_iter: DataLoader
    """
    batch_size = config.batch_size
    X_ids = change_sentence_to_ids(args, X, word2id)

    X_ids = padding_or_truncate(X_ids, padding, config.max_seq_length)

    e_ids = change_emoji_to_ids(e, emoji2id)

    batch_count = int(len(X) / batch_size)
    batch_X, batch_y, batch_emoji = [], [], []

    for i in range(batch_count):
        batch_X.append(X_ids[i * batch_size: (i + 1) * batch_size])
        batch_y.append(y[i * batch_size: (i + 1) * batch_size])
        batch_emoji.append(e_ids[i * batch_size: (i + 1) * batch_size])

    return batch_X, batch_emoji, batch_y, batch_count

    # X_ids = torch.LongTensor(X_ids)
    # y_ids = torch.LongTensor(y)
    #
    # dataset = Data.TensorDataset(X_ids, y_ids)
    #
    # if data_mode == 'train':
    #     data_iter = Data.DataLoader(dataset, config.batch_size, shuffle=True)
    # else:
    #     data_iter = Data.DataLoader(dataset, config.batch_size, shuffle=False)
    #
    # return data_iter


def evaluate(model, batch_X, batch_e, batch_y, batch_count, device):
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
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for i in range(batch_count):
            X = batch_X[i]
            e = batch_e[i]

            labels = torch.LongTensor(batch_y[i]).to(device)
            outputs = model(X, e)

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
    return loss_total / batch_count, acc, macro_P, macro_R, macro_F1


def train(device):

    if args.mode == 'Weibo':
        train_x, train_e, train_y = load_data(
            os.path.join(fp.weibo_data, 'train.csv'))
        dev_x, dev_e, dev_y = load_data(
            os.path.join(fp.weibo_data, 'dev.csv'))
        test_x, test_e, test_y = load_data(
            os.path.join(fp.weibo_data, 'test.csv'))
        w2v_embed = utils.read_file('pkl', fp.chinese_word2vec)
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'weibo_word2id.json'))
        emoji2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'emoji2id.json'))

        texts = []
        texts.extend(train_x)
        texts.extend(dev_x)
        texts.extend(test_x)

        words_set = get_wordset_and_lables(texts)
        embed = get_all_words_embed(w2v_embed, words_set)

        emoji2def_csv = pd.read_csv(fp.weibo_emoji2def)
        num_emojis = len(emoji2def_csv)
    else:
        train_x, train_e, train_y = load_data(
            os.path.join(fp.github_data, 'train.csv'))
        dev_x, dev_e, dev_y = load_data(
            os.path.join(fp.github_data, 'dev.csv'))
        test_x, test_e, test_y = load_data(
            os.path.join(fp.github_data, 'test.csv'))
        w2v_embed = KeyedVectors.load_word2vec_format(fp.english_word2vec, binary=True)
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'github_word2id.json'))
        emoji2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'github_emoji2id.json'))

        words_set = list(word2id.keys())
        embed = get_all_words_embed(w2v_embed, words_set)

        emoji2def_csv = pd.read_csv(fp.github_emoji2def)
        num_emojis = len(emoji2def_csv)

    padding = word2id['<PAD>']

    config = WATT_BiE_LSTM.Config(num_emojis=num_emojis)
    # config.lr = 1e-5
    # config.batch_size = 64

    # train_iter = get_data_iter(args, train_x, train_y, word2id, padding, config, 'train')
    # dev_iter = get_data_iter(args, dev_x, dev_y, word2id, padding, config, 'dev')
    # test_iter = get_data_iter(args, test_x, test_y, word2id, padding, config, 'test')

    train_batch_X, train_batch_e, train_batch_y, train_batch_count = get_data_iter(args, train_x,
                                                                                   train_e, train_y,
                                                                                   word2id, emoji2id,
                                                                                   padding, config)
    dev_batch_X, dev_batch_e, dev_batch_y, dev_batch_count = get_data_iter(args, dev_x, dev_e, dev_y,
                                                                           word2id, emoji2id, padding,
                                                                           config)
    test_batch_X, test_batch_e, test_batch_y, test_batch_count = get_data_iter(args, test_x, test_e,
                                                                               test_y, word2id, emoji2id,
                                                                               padding, config)

    if args.model == 'WATT':
        save_path = os.path.join(fp.model_parameters, 'watt_bie_lstm.bin')
        model = WATT_BiE_LSTM.Model(embed, config, device).to(device)
    else:
        save_path = os.path.join(fp.model_parameters, 'matt_bie_lstm.bin')
        model = MATT_BiE_LSTM.Model(embed, config, device).to(device)

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
        for idx in range(train_batch_count):
            # X = X.to(device)
            # y = y.to(device)
            X = train_batch_X[idx]
            e = train_batch_e[idx]

            y = torch.LongTensor(train_batch_y[idx]).to(device)
            y_hat = model(X, e)

            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                pred = torch.max(y_hat.data.cpu(), 1)[1].numpy()
                acc = accuracy_score(y.detach().cpu().numpy(), pred)

                dev_loss, dev_acc, dev_macro_P, dev_macro_R, dev_macro_F1 = evaluate(model, dev_batch_X, dev_batch_e,
                                                                                     dev_batch_y, dev_batch_count,
                                                                                     device)

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

    if args.model == 'WATT':
        model = WATT_BiE_LSTM.Model(embed, config, device).to(device)
    else:
        model = MATT_BiE_LSTM.Model(embed, config, device).to(device)
    model.load_state_dict(torch.load(save_path))

    test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1 = evaluate(model, test_batch_X, test_batch_e,
                                                                              test_batch_y, test_batch_count, device)
    print(args.model)
    print('Test loss %f | Test acc %f | Test macro P %f | Test macro R %f | Test macro F1 %f' % (
        test_loss, test_acc, test_macro_P, test_macro_R, test_macro_F1
    ))


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--mode',
                    help='Github, Weibo',
                    default='Weibo')
parser.add_argument('--model',
                    help='MATT, WATT',
                    default='MATT')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(device)