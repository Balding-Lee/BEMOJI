"""
Transformer for pre-training
:author: Qizhi Li
"""
import os
import sys
import argparse
import torch
from torch.optim import AdamW
import time
import warnings
from transformers import get_cosine_schedule_with_warmup

sys.path.append('..')
from static_data import file_path as fp
from models import bert_mlm


def load_data(mode):
    """
    加载预训练数据集
    :return:
    """
    texts = []

    if mode == 'pre_train_chinese':
        with open(fp.pre_train_chinese_data, encoding='utf-8') as f:
            content = f.readlines()
    elif mode == 'pre_train_chinese':
        with open(fp.pre_train_english_data, encoding='utf-8') as f:
            content = f.readlines()

    for line in content:
        split_line = line.rstrip('\n').split('\t')
        texts.append(split_line[0])

    return texts


def get_data_iter(batch_size, mode):
    """
    分 batch 获得数据
    :param batch_size: int
    :param mode: str
            'pre_train_chinese': 加载中文数据集
            'pre_train_english': 加载英文数据集
    :return inputs_batches: list
    :return batch_count: int
    """
    texts = load_data(mode)
    batch_count = int(len(texts) / batch_size)
    inputs_batches = []

    for i in range(batch_count):
        inputs_batches.append(texts[i * batch_size: (i + 1) * batch_size])

    return inputs_batches, batch_count


def train(args, device):
    # 判断是否有 checkpoint, 防止系统中断而需要重新训练
    checkpoint_epoch = 0
    files = os.listdir(fp.pre_train_parameters)
    for file in files:
        if args.mode == 'pre_train_chinese':
            if args.model in file and 'github' not in file:
                epoch_tmp = int(file.rstrip('.bin').split('_')[-1])
                if epoch_tmp > checkpoint_epoch:
                    checkpoint_epoch = epoch_tmp
        elif args.mode == 'pre_train_english':
            if args.model in file and 'github' in file:
                epoch_tmp = int(file.rstrip('.bin').split('_')[-1])
                if epoch_tmp > checkpoint_epoch:
                    checkpoint_epoch = epoch_tmp

    config = bert_mlm.PreTrainConfig()
    config.mlm_config()
    if args.mode == 'pre_train_chinese':
        if args.model == 'BERT_base':
            from_path = '../static_data/bert-base-chinese'
        elif args.model == 'RoBERTa_base':
            from_path = '../static_data/chinese-roberta-wwm-ext'
        elif args.model == 'RoBERTa_large':
            from_path = '../static_data/chinese-roberta-wwm-ext-large'
        elif args.model == 'DistilBert_base':
            from_path = '../static_data/distilbert-base-zh-cased'

        config.pre_train_config(batch_size=64, epochs=10,
                                learning_rate=1e-5,
                                weight_decay=1e-4,
                                from_path=from_path,
                                device=device)
    elif args.mode == 'pre_train_english':
        if args.model == 'BERT_base':
            from_path = '../static_data/bert-base-uncased'
        elif args.model == 'BERT_large':
            from_path = '../static_data/bert-large-uncased'
        elif args.model == 'RoBERTa_base':
            from_path = '../static_data/roberta-base'
        elif args.model == 'RoBERTa_large':
            from_path = '../static_data/roberta-large'
        elif args.model == 'DistilBert_base':
            from_path = '../static_data/distilbert-base-uncased'

        config.pre_train_config(batch_size=128, epochs=10,
                                learning_rate=1e-4,
                                weight_decay=1e-4,
                                from_path=from_path,
                                device=device)

    input_batches, batch_count = get_data_iter(config.batch_size, args.mode)

    # 加载存储点, 如果有存储文件的话
    if checkpoint_epoch > 0:
        if args.mode == 'pre_train_chinese':
            parameter_path = os.path.join(fp.pre_train_parameters,
                                          '{}_ep_{}.bin'.format(args.model,
                                                                checkpoint_epoch))
            print('loaded pretrained parameters')
        elif args.mode == 'pre_train_english':
            parameter_path = os.path.join(fp.pre_train_parameters,
                                          '{}_github_ep_{}.bin'.format(args.model,
                                                                       checkpoint_epoch))
        else:
            parameter_path = None
        epochs = range(checkpoint_epoch, config.epochs)
    else:
        parameter_path = None
        epochs = range(config.epochs)

    pretrain_model = bert_mlm.TransformerMLM(args, config, device, parameter_path)

    optimizer = AdamW(pretrain_model.parameters(),
                      lr=config.learning_rate,
                      weight_decay=config.weight_decay)
    schedule = get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=len(input_batches),
                                               num_training_steps=config.epochs * len(input_batches))

    total_time = 0

    print('start pretraining')
    for epoch in epochs:
        index = 0
        loss_sum = 0
        pretrain_model.train()
        start = time.time()
        for i in range(batch_count):
            input_seqs = input_batches[i]

            loss = pretrain_model(input_seqs)
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

            index += 1
            if (index + 1) % 100 == 0:
                print("Epoch {:02d} | Step {:04d}/{:04d} | Loss {:04f} | Time {:.0f}".format(
                    epoch + 1, index + 1, batch_count, loss_sum / (index + 1), time.time() - start))

        end = time.time()
        print("epoch {}, Loss {:04f}, duration {:04f}".format(
            epoch + 1, loss_sum / batch_count, end - start))
        total_time += end - start

        if args.mode == 'pre_train_chinese':
            torch.save(pretrain_model.context_bert.state_dict(),
                       os.path.join(fp.pre_train_parameters,
                                    '{}_ep_{}.bin'.format(args.model,
                                                          epoch + 1)))
        elif args.mode == 'pre_train_english':
            torch.save(pretrain_model.context_bert.state_dict(),
                       os.path.join(fp.pre_train_parameters,
                                    '{}_github_ep_{}.bin'.format(args.model,
                                                                 epoch + 1)))

    print('total pretraining time: ', total_time)


warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model',
                    help='BERT_base, BERT_large, RoBERTa_base, RoBERTa_large, '
                         'DistilBert_base. Note: only english dataset has BERT_large')
parser.add_argument('--mode',
                    help='pre_train_english: fine tune English dataset,'
                         'pre_train_chinese: fine tune Chinese dataset',
                    default='pre_train_chinese')
args = parser.parse_args()
train(args, device)
