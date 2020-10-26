import argparse
import pandas as pd
import random
import torch
import torch.nn as nn
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
# from model import BatchProgramClassifier
from encoder.astnn import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        # data.append(item[1])
        # labels.append(item[2]-1)
        data.append(item[2])
        labels.append(item[0]-1)
    return data, torch.LongTensor(labels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('poj_a')
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--classes', default=293, type=int, metavar='N', help='number of program classes')
    parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    # utils
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

    args = parser.parse_args()

    root = 'data/poj_cls/'
    train_data = pd.read_pickle(root+'train/blocks.pkl')
    val_data = pd.read_pickle(root + 'dev/blocks.pkl')
    test_data = pd.read_pickle(root+'test/blocks.pkl')

    word2vec = Word2Vec.load(root+"train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 293
    # args.epochs = 15
    EPOCHS = 10
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    if args.resume is not '':
        from pretext import build_model
        from collections import namedtuple
        import json
        with open(args.resume + '/args.json', 'r') as f:
            args_dict = json.load(f)
        # https://blog.csdn.net/fuli911/article/details/109178453
        Argument = namedtuple('Argument', args_dict)
        args_h = Argument(**args_dict)
        model_h = build_model(args_h)
        checkpoint = torch.load(args.resume + '/model_last.pth')
        model_h.load_state_dict(checkpoint['state_dict'])
        print('Loaded from: {}'.format(args.resume))
        print(model_h.encoder_q0.base_encoder)
        model = nn.Sequential(
            model_h.encoder_q0.base_encoder,
            nn.Linear(128, args.classes)
        ).cuda()
    else:
        model = BatchProgramClassifier(
            EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,
            ENCODE_DIM, args.classes, args.batch_size,
            USE_GPU, embeddings)
        if USE_GPU:
            model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(args.epochs):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, args.batch_size)
            i += args.batch_size
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            if type(model).__name__ == 'Sequential':
                model.hidden = model[0].init_hidden()
            else:
                model.hidden = model.init_hidden()
            # print(train_inputs)
            # assert 0
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(val_data):
            batch = get_batch(val_data, i, args.batch_size)
            i += args.batch_size
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            if type(model).__name__ == 'Sequential':
                model.hidden = model[0].init_hidden()
            else:
                model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, args.epochs, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        batch = get_batch(test_data, i, args.batch_size)
        i += args.batch_size
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        if type(model).__name__ == 'Sequential':
            model.hidden = model[0].init_hidden()
        else:
            model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)