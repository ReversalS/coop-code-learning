# here defines the pretext tasks
# current ideas:
# 1. sets using homogeneous encoders
# 2. combine heterogeneous encoders (requires design of contrast)
# 3. ...
from datetime import datetime
import argparse
import os
import tqdm
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler


# ver 0
from contrast.hetero import ModelHeteroMoCo
from encoder.astnn import ASTnnEncoder
from encoder.c2v import Code2VecEncoder
from encoder.base import ModelWithProjection
from dataloader import DataLoader


def adjust_learning_rate(optimizer, epoch, args):
    "Decay the learning rate based on schedule"
    lr = args.lr
    if args.cos:    # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:   # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(net, dataloader, train_optimizer, epoch, args):

    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num = 0.0, 0

    for x1, x2 in dataloader:
        if x2.shape[0] != args.batch_size:
            continue

        loss = net(x1, x2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
    print('Train Epoch: [{} / {}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num


def build_model(args):
    "TODO: dynamic model definition"
    encoder_q0 = ModelWithProjection(
        ASTnnEncoder(128, 100, 6721, 128, args.batch_size, True, None), 
        out_dim=128)
    encoder_k0 = ModelWithProjection(
        ASTnnEncoder(128, 100, 6721, 128, args.batch_size, True, None), 
        out_dim=128)
    encoder_q1 = ModelWithProjection(
        Code2VecEncoder(30002, 50002, 1, 128, 128, 128, 600, 0, 0.1), 
        out_dim=128)
    encoder_k1 = ModelWithProjection(
        Code2VecEncoder(30002, 50002, 1, 128, 128, 128, 600, 0, 0.1), 
        out_dim=128)
    model = ModelHeteroMoCo(
        encoder_q0, encoder_k0, encoder_q1, encoder_k1, 
        dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t,
        bn_splits=args.bn_splits)
    return model.cuda()


if True:
    parser = argparse.ArgumentParser(description='Pretext on c programs')

    # lr: 0.06 for batch 512 (or 0.03 for batch 256)
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

    parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

    parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

    # utils
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

    args = parser.parse_args()

    if args.results_dir == '':
        args.results_dir = './results/cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

    print(args)

    model = build_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    epoch_start = 1
    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        opimitzer.load_state_dict([checkpoint['optimizer']])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))
    
    # logging
    results = {'train_loss': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    train_loader = DataLoader('train', batch_size=args.batch_size)

    for epoch in range(epoch_start, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch+1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')
        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }, args.results_dir + '/model_last.pth')



