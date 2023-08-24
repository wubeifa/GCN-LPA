import argparse
import numpy as np
import tensorflow as tf
from time import time
from data_loader import load_data, load_npz, load_random
from train import train


def switch_case(argument):
    if argument == 1:
        # 执行操作1:cora
        parser.add_argument('--dataset', type=str, default='cora', help='which dataset to use')
        parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
        parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
        parser.add_argument('--gcn_layer', type=int, default=5, help='number of GCN layers')
        parser.add_argument('--lpa_iter', type=int, default=5, help='number of LPA iterations')
        parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
        parser.add_argument('--lpa_weight', type=float, default=10, help='weight of LP regularization')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
        parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    elif argument == 2:
        # 执行操作2:citeseer
        parser.add_argument('--dataset', type=str, default='citeseer', help='which dataset to use')
        parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
        parser.add_argument('--dim', type=int, default=16, help='dimension of hidden layers')
        parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
        parser.add_argument('--lpa_iter', type=int, default=5, help='number of LPA iterations')
        parser.add_argument('--l2_weight', type=float, default=5e-4, help='weight of l2 regularization')
        parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
        parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
        parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
    elif argument == 3:
        # 执行操作3:pubmed
        parser.add_argument('--dataset', type=str, default='pubmed', help='which dataset to use')
        parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
        parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
        parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
        parser.add_argument('--lpa_iter', type=int, default=1, help='number of LPA iterations')
        parser.add_argument('--l2_weight', type=float, default=2e-4, help='weight of l2 regularization')
        parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
        parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
        parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    elif argument == 4:
        # 执行操作4:coauthor-cs
        parser.add_argument('--dataset', type=str, default='coauthor-cs', help='which dataset to use')
        parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
        parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
        parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
        parser.add_argument('--lpa_iter', type=int, default=2, help='number of LPA iterations')
        parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
        parser.add_argument('--lpa_weight', type=float, default=2, help='weight of LP regularization')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
        parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    elif argument == 5:
        # 执行操作5:coauthor-phy
        parser.add_argument('--dataset', type=str, default='coauthor-phy', help='which dataset to use')
        parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
        parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
        parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
        parser.add_argument('--lpa_iter', type=int, default=3, help='number of LPA iterations')
        parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
        parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
        parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    else:
        # 无效的选择
        print("无效的数据集选择")

seed = 234
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()

switch_case(2)  

t = time()
args = parser.parse_args()

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    data = load_data(args.dataset)
elif args.dataset in ['coauthor-cs', 'coauthor-phy']:
    data = load_npz(args.dataset)
else:
    n_nodes = 1000
    data = load_random(n_nodes=n_nodes, n_train=100, n_val=200, p=10/n_nodes)

train(args, data)
print(args)
print('time used: %d s' % (time() - t))
