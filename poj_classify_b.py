import argparse
import pickle
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder.c2v import Code2VecEncoder, PathContextEncoder, SetPathContextEncoder


def get_batch(data, labels, idx, batch_size, max_context=600, pad_idx=0):
    x = data[idx: idx + batch_size]
    y = labels[idx: idx + batch_size]
    # reform x
    x_r = []
    for c in x:
        c = c[:max_context] # trimming (consider other strategies)
        c += [[pad_idx, pad_idx, pad_idx]] * (max_context - len(c)) # padding
        x_r.append(c)
    # reform y
    y = [l - 1 for l in y]
    return torch.LongTensor(x_r), torch.LongTensor(y)
    # return data, labels


def train(model, train_data, train_labels, optimizer, loss_function, epoch, args):
    model.train()
    # adjust_learning_rate()
    total_acc, total_loss, total_num = 0.0, 0.0, 0
    i = 0
    while i < len(train_data):
        x, y = get_batch(train_data, train_labels, i, args.batch_size)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        # print('max left token id:', torch.max(x[:, :, 0]),
        #     'max path id:', torch.max(x[:, :, 1]),
        #     'max right token id:', torch.max(x[:, :, 2])
        # )
        i += args.batch_size

        output = model(x)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == y).sum()
        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size

    return total_loss / total_num, total_acc / total_num


def validate(model, val_data, val_labels, loss_function, args):
    model.eval()
    total_acc, total_loss, total_num = 0.0, 0.0, 0
    i = 0
    while i < len(val_data):
        x, y = get_batch(val_data, val_labels, i, args.batch_size)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        i += args.batch_size

        output = model(x)
        loss = loss_function(output, y)
        
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == y).sum()
        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
    
    return total_loss / total_num, total_acc / total_num


def test(model, test_data, test_labels, loss_function, args):
    model.eval()
    total_acc, total_loss, total_num = 0.0, 0.0, 0
    i = 0
    while i < len(test_data):
        x, y = get_batch(test_data, test_labels, i, args.batch_size)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        i += args.batch_size

        output = model(x)
        loss = loss_function(output, y)
        
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == y).sum()
        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
    
    return total_loss / total_num, total_acc / total_num


if __name__ == '__main__':

    parser = argparse.ArgumentParser('poj_b')
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--classes', default=293, type=int, metavar='N', help='number of program classes')
    parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--token-vocab', default=30000 + 2, type=int, metavar='N', help='token vocabulary size')
    parser.add_argument('--path-vocab', default=50000 + 2, type=int, metavar='N', help='path vocabulary size')
    parser.add_argument('--contexts', default=600, type=int, metavar='N', help='number of path contexts for input')
    # utils
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

    args = parser.parse_args()

    root = 'data/poj_cls/'
    with open(root + 'train/data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(root + 'train/labels.pkl', 'rb') as f:
        train_labels = pickle.load(f)
    with open(root + 'dev/data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open(root + 'dev/labels.pkl', 'rb') as f:
        val_labels = pickle.load(f)
    with open(root + 'test/data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(root + 'test/labels.pkl', 'rb') as f:
        test_labels = pickle.load(f)
    # train_data = torch.randint(1, 20000, (64, 600, 3))
    # train_labels = torch.randint(0, 292, (64, ))

    
    # print(model)

    if args.resume is not '':
        from pretext import build_model
        from collections import namedtuple
        import json
        with open(args.results_dir + '/args.json', 'r') as f:
            args_dict = json.load(f)
        # https://blog.csdn.net/fuli911/article/details/109178453
        Argument = namedtuple('Argument', args_dict)
        args_h = Argument(**args_dict)
        model_h = build_model(args_h)
        checkpoint = torch.load(args_h.resume)
        model_h.load_state_dict(checkpoint['state_dict'])
        opimitzer.load_state_dict([checkpoint['optimizer']])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))
        model = nn.Sequential(
            model_h.encoder_q1.base_encoder,
            nn.Linear(128, args.classes)
        )
    else:
        model = nn.Sequential(
            Code2VecEncoder(args.token_vocab, args.path_vocab, 1,
                128, 128, 128, 600, 0, 0.1),
            # PathContextEncoder(
            #     args.token_vocab, args.path_vocab, 128, 128, 600, 0, 0.1),
            # SetPathContextEncoder(
            #     args.token_vocab, args.path_vocab,
            #     token_dim=128, path_dim=128, max_len=args.contexts, pad_idx=0, dropout=0.1,
            # num_outputs=1, num_inds=64, num_heads=4, ln=True),
            nn.Linear(128, args.classes)
        ).cuda()
    
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model = model
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(
            model, train_data, train_labels, optimizer, loss_function,
            epoch, args
        )
        val_loss, val_acc = validate(
            model, val_data, val_labels, loss_function, args
        )
        if val_acc > best_acc:
            best_model = model
            best_acc = val_acc
        end_time = time.time()
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
              % (epoch + 1, args.epochs, train_loss, val_loss,
                 train_acc, val_acc, end_time - start_time))
    test_loss, test_acc = test(
        best_model, test_data, test_labels, loss_function, args
    )
    print('Testing Loss: %.3f, Testing Acc: %.3f' % (test_loss, test_acc))
