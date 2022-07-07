"""
DeepMoji for pre-training
:author: Qizhi Li
"""
import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pyprind

sys.path.append('..')
import utils
from static_data import file_path as fp
from models import DeepMoji


def load_data(args, emoji2def):
    """
    :param args: Object
    :param emoji2def: dict
            {emoji: definition}
    :return texts: list
    :return emojis: list
    """
    texts, emojis = [], []
    if args.mode == 'pre_train_chinese':

        with open(fp.pre_train_chinese_data, 'r', encoding='utf-8') as f:
            content = f.readlines()

        for line in content:
            split_data = line.rstrip('\n').split('\t')
            text, emojis_in_text = split_data[0], split_data[1]

            emojis_set = set(emojis_in_text.split(','))
            # If one sentence has multiple emojis, convert it into one sentence with one emoji
            for emoji in emojis_set:
                if emoji2def.__contains__(emoji):
                    texts.append(text)
                    emojis.append(emoji)
    else:
        with open(fp.pre_train_english_data, 'r', encoding='utf-8') as f:
            content = f.readlines()

        for line in content:
            split_data = line.rstrip('\n').split('\t')

            texts.append(split_data[0])
            emojis.append(split_data[1])

    return texts, emojis


def get_wordset_and_lables(args, emojis):
    """
    Obtaining word set and labels
    :param args: Object
    :param emojis: list
    :return word_set: set
    :return labels: list
    """

    if args.mode == 'pre_train_chinese':
        emoji2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'emoji2id.json'))
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'word2id.json'))
    else:
        emoji2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'github_emoji2id.json'))
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'github_word2id.json'))
    labels = []

    for emoji in emojis:
        labels.append(emoji2id[emoji])

    word_set = list(word2id.keys())

    return word_set, labels


def get_all_words_embed(w2v_embed, words_set):
    """
    Obtaining word embedding
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

    # '<PAD>' embedding is all 0
    embed = torch.cat((embed, torch.zeros(1, embed.shape[1])))

    return embed


def change_sentence_to_ids(args, dataset, word2id):
    """
    Convert sentence into ids
    :param args: Object
    :param dataset: list
    :param word2id: dict
            {word: id}
    :return: sentences_ids: list
    """
    sentences_ids = []
    for sentence in dataset:
        if args.mode == 'pre_train_chinese':
            words = list(sentence)
        else:
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
            The id of '<PAD>'
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


def get_data_iter(args, texts, labels, config):
    """
    Package batch
    :param args: Object
    :param texts: list
    :param labels: list
    :param config: Object
    :return train_iter: DataLoader
    :return dev_iter: DataLoader
    """
    if args.mode == 'pre_train_chinese':
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'word2id.json'))
    else:
        word2id = utils.read_file('json', os.path.join(fp.deepmoji_vocab, 'github_word2id.json'))
    padding = word2id['<PAD>']
    train_X, dev_X, train_y, dev_y = train_test_split(texts, labels, test_size=0.1,
                                                      stratify=labels)

    train_X = change_sentence_to_ids(args, train_X, word2id)
    dev_X = change_sentence_to_ids(args, dev_X, word2id)

    train_X = padding_or_truncate(train_X, padding, config.max_seq_length)
    dev_X = padding_or_truncate(dev_X, padding, config.max_seq_length)

    train_X = torch.LongTensor(train_X)
    dev_X = torch.LongTensor(dev_X)
    train_y = torch.LongTensor(train_y)
    dev_y = torch.LongTensor(dev_y)

    train_dataset = Data.TensorDataset(train_X, train_y)
    dev_dataset = Data.TensorDataset(dev_X, dev_y)

    train_iter = Data.DataLoader(train_dataset, config.batch_size, shuffle=True)
    dev_iter = Data.DataLoader(dev_dataset, config.batch_size)

    return train_iter, dev_iter


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
    return loss_total / len(data_iter), acc


def train(args, device):

    if args.mode == 'pre_train_chinese':
        emoji2def = pd.read_csv(fp.weibo_emoji2def)
        emoji_defs = {}
        for _, line in emoji2def.iterrows():
            emoji_defs[line['微博表情']] = line['表情定义']

        w2v_embed = utils.read_file('pkl', fp.chinese_word2vec)
    elif args.mode == 'pre_train_english':
        emoji2def = pd.read_csv(fp.github_emoji2def)
        emoji_defs = {}
        for _, line in emoji2def.iterrows():
            emoji_defs[line['emoji']] = line['definition']

        w2v_embed = KeyedVectors.load_word2vec_format(fp.english_word2vec, binary=True)

    texts, emojis = load_data(args, emoji_defs)

    words_set, labels = get_wordset_and_lables(args, emojis)
    embed = get_all_words_embed(w2v_embed, words_set)
    config = DeepMoji.Config(num_classes=len(emoji_defs))
    
    train_iter, dev_iter = get_data_iter(args, texts, labels, config)

    model = DeepMoji.DeepMoji(embed, config, device).to(device)
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

                dev_loss, dev_acc = evaluate(model, dev_iter, device)

                model.train()

                print('epoch %d | iter %d | train loss %f | train accuracy %f | '
                      'dev loss %f | dev accuracy %f | duration %0f' % (
                          epoch + 1, i + 1, l.item(), acc, dev_loss, dev_acc, time.time() - iter_start
                      ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # torch.save(model.state_dict(), save_path)
                    # torch.save(model.embeddings.state_dict(),
                    #            os.path.join(fp.pre_train_parameters, 'deepmoji_embed.bin'))
                    if args.mode == 'pre_train_chinese':
                        torch.save(model.lstm_0.state_dict(),
                                   os.path.join(fp.pre_train_parameters, 'deepmoji_lstm0.bin'))
                        torch.save(model.lstm_1.state_dict(),
                                   os.path.join(fp.pre_train_parameters, 'deepmoji_lstm1.bin'))
                        torch.save(model.attention.state_dict(),
                                   os.path.join(fp.pre_train_parameters, 'deepmoji_attention.bin'))
                    else:
                        torch.save(model.lstm_0.state_dict(),
                                   os.path.join(fp.pre_train_parameters, 'deepmoji_github_lstm0.bin'))
                        torch.save(model.lstm_1.state_dict(),
                                   os.path.join(fp.pre_train_parameters, 'deepmoji_github_lstm1.bin'))
                        torch.save(model.attention.state_dict(),
                                   os.path.join(fp.pre_train_parameters, 'deepmoji_github_attention.bin'))
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


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--mode',
                    help='pre_train_english: fine tune English dataset,'
                         'pre_train_chinese: fine tune Chinese dataset',
                    default='pre_train_chinese')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
train(args, device)
