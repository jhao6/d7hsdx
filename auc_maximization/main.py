import copy
import json
from random import shuffle
from collections import Counter
import torch
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from reptile import Learner
from methods import aid, saba, ma_soba, ttsa, bo_rep, accbo, sustain, vrbo

from data_loader import Sent140Dataset,  collate_pad
from torch.utils.data import DataLoader

import random
import numpy as np
torch.backends.cudnn.enabled = False
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def ImbalanceGenerator(data, imratio=0.2, split=True):
    val = copy.deepcopy(data)
    X = data.sentences
    Y = data.labels
    id_list = list(range(len(X)))
    np.random.shuffle(id_list)
    X = [X[id] for id in id_list]
    Y = [Y[id] for id in id_list]
    X_copy = X.copy()
    Y_copy = Y.copy()
    num_neg = np.where(np.array(Y_copy) == 0)[0][:1000].shape[0]
    num_pos = np.where(np.array(Y_copy) == 1)[0].shape[0]
    keep_num_pos = int((imratio / (1 - imratio)) * num_neg)
    neg_id_list = np.where(np.array(Y_copy) == 0)[0][:1000]
    pos_id_list = np.where(np.array(Y_copy) == 1)[0][:keep_num_pos]
    remain_list = [neg_id_list.tolist() + pos_id_list.tolist()][0]
    X_copy = [X_copy[i] for i in remain_list]
    Y_copy =  [Y_copy[i] for i in remain_list]
    id_list = list(range(len(X_copy)))
    np.random.shuffle(id_list)
    sentences =  [X_copy[id] for id in id_list]
    labels = [Y_copy[id] for id in id_list]
    size_data = len(labels)
    if split:
        data.sentences = sentences[:int(size_data/2)]
        val.sentences = sentences[int(size_data/2):]
        data.labels = labels[:int(size_data/2)]
        val.labels = labels[int(size_data/2):]
        return data, val
    else:
        data.sentences = sentences
        data.labels = labels
        return data

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default='sentment140', type=str,
                        help="dataset: [news_data, snli, sentment140]", )

    parser.add_argument("--data_path", default='data', type=str,
                        help="Path to dataset file")

    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch_size", )


    parser.add_argument("--save_direct", default='./', type=str,
                        help="Path to save file")

    parser.add_argument("--word2vec", default="data/wordvec.pkl", type=str,
                        help="Path to word2vec file")

    parser.add_argument("--methods" , default='saba', type=str,
                        help="choise method [aid, ttsa, saba, ma-soba, bo-rep, sustain, vrbo, accbo]")

    parser.add_argument("--num_labels", default=2, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=25, type=int,
                        help="Number of outer interation")
    
    parser.add_argument("--k_spt", default=20, type=int,
                        help="Number of support samples per task")
    
    parser.add_argument("--k_qry", default=20, type=int,
                        help="Number of query samples per task")

    parser.add_argument("--outer_batch_size", default=20, type=int,
                        help="Batch of task size")
    
    parser.add_argument("--inner_batch_size", default=32, type=int,
                        help="Training batch size in inner iteration")


    parser.add_argument("--hessian_lr", default=1e-2, type=float,
                        help="update for hessian")

    parser.add_argument("--hessian_q", default=5, type=int,
                        help="Q steps for hessian-inverse-vector product")

    parser.add_argument("--outer_update_lr", default= 1e-1, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=1e-1, type=float,
                        help="Inner update learning rate")
    
    parser.add_argument("--inner_update_step", default=64, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--outer_update_step", default=1, type=int,
                        help="Number of interation in the outer loop during train time")

    parser.add_argument("--inner_update_step_eval", default=40, type=int,
                        help="Number of interation in the inner loop during test time")
    
    parser.add_argument("--num_task_train", default=400, type=int,
                        help="Total number of meta tasks for training")
    
    parser.add_argument("--num_task_test", default=50, type=int,
                        help="Total number of tasks for testing")

    parser.add_argument("--grad_clip", default=False, type=bool,
                        help="whether grad clipping or not")

    parser.add_argument("--grad_normalized", default=True, type=bool,
                        help="whether grad normalized or not")

    parser.add_argument("--gamma", default=1e-3, type=float,
                        help="clipping threshold")

    parser.add_argument("--no_meta", default=False, type=bool,
                        help="whether use meta learning")

    parser.add_argument("--seed", default=2, type=int,
                        help="random seed")

    # single loops parameters
    parser.add_argument("--beta", default=0.90, type=float,
                        help="momentum parameters")

    parser.add_argument("--nu", default=1e-2, type=float,
                        help="learning rate of z")

    parser.add_argument("--xi", default=0.08, type=float,
                        help="step-size ratio")

    parser.add_argument("--y_warm_start", default=3, type=int,
                        help="update steps for y")

    parser.add_argument("--interval", default=2, type=int,
                        help="update interval for y")

    parser.add_argument("--spider_loops", default=3, type=int,
                        help="the spider loops for vrbo")


    parser.add_argument("--update_interval", default=5, type=int,
                        help="update interval for vrbo")

    parser.add_argument("--lamb", default=1e-1, type=float,
                        help="lagerange coefficient")

    parser.add_argument("--imratio", default=0.8, type=float,
                        help="The ratio of imbalance")

    # RNN hyperparameter settings
    parser.add_argument("--word_embed_dim", default=300, type=int,
                        help="word embedding dimensions")

    parser.add_argument("--encoder_dim", default=4096, type=int,
                        help="encodding dimensions")

    parser.add_argument("--n_enc_layers", default=2, type=int,
                        help="encoding layers")

    parser.add_argument("--fc_dim", default=1024, type=int,
                        help="dimension of fully-connected layer")

    parser.add_argument("--n_classes", default=2, type=int,
                        help="classes of targets")

    parser.add_argument("--linear_fc", default=False, type=bool,
                        help="classes of targets")

    parser.add_argument("--pool_type", default="max", type=str,
                        help="type of pooling")

    parser.add_argument("--noise_rate", default=0.0, type=float,
                        help="rate for label noise")

    args = parser.parse_args()
    random_seed(args.seed)
    if args.data == 'sentment140':
        if os.path.isfile(f'data/train_data_{args.imratio}'):
            print('loading data...')
            train = torch.load(f'data/train_data_{args.imratio}')
            val = torch.load(f'data/val_data_{args.imratio}')
            test = torch.load(f'data/test_data_{args.imratio}')
        else:
            trainset = Sent140Dataset("../data", "train", noise_rate=args.noise_rate)
            train, val = ImbalanceGenerator(trainset, imratio=args.imratio)
            torch.save(train, f'data/train_data_{args.imratio}')
            torch.save(val, f'data/val_data_{args.imratio}')
            test = Sent140Dataset("../data", "test")
            # test = ImbalanceGenerator(testset, imratio=args.imratio, split=False)
            torch.save(test, f'data/test_data_{args.imratio}')
        # test = ImbalanceGenerator(testset, imratio=0.2)
        args.n_labels = 2
        args.n_classes = 2
        training_size = train.__len__()

    else:
        print('Do not support this data')

    st = time.time()

    if args.data == 'news_data':
        args.n_classes=2
        args.n_labels=2


    if args.methods == 'aid':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 1e-3
        args.hessian_lr = 1e-2
        args.grad_clip = False
        # args.gamma = 1.0
        args.inner_update_step = 3
        args.hessian_q= 3
        # args.inner_batch_size = args.num_task_test
        learner = aid.Learner(args, training_size)

    if args.methods == 'ttsa':
        args.outer_update_lr = 5e-3
        args.inner_update_lr = 1e-2
        args.hessian_lr = 1e-1
        args.grad_clip = False
        args.y_warm_start = 1
        args.inner_update_step = 3
        # args.inner_batch_size = args.num_task_test
        learner = ttsa.Learner(args, training_size)


    elif args.methods == "saba":
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 5e-3
        args.grad_normalized = False
        args.grad_clip = False
        args.beta = 0.0
        args.nu = 1e-2
        args.inner_update_step = 1
        # args.inner_batch_size = args.num_task_test
        learner = saba.Learner(args, training_size)

    elif args.methods == 'ma-soba':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 5e-3
        args.grad_normalized = False
        args.grad_clip = False
        args.beta = 0.9
        args.nu = 1e-1
        args.inner_update_step = 1
        # args.inner_batch_size = args.num_task_test
        learner = ma_soba.Learner(args, training_size)

    elif args.methods == 'bo-rep':
        args.outer_update_lr = 1e-3
        args.inner_update_lr = 1e-3
        args.grad_normalized =  True
        args.grad_clip = False
        # args.gamma = 1.0e-2
        # args.y_warm_start = 10
        args.xi = 1
        args.beta = 0.9
        args.nu = 1e-2
        args.y_warm_start = 3
        args.inner_update_step = 3
        args.interval = 2
        # args.inner_batch_size = args.num_task_test
        learner = bo_rep.Learner(args, training_size)


    elif args.methods == 'sustain':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 5e-3
        args.hessian_lr = 1e-2
        args.grad_normalized =  True
        args.grad_clip = False
        args.hessian_q = 3
        # args.y_warm_start = 10
        args.gamma = 0.6
        args.beta = 0.9
        # args.alpha = 0.0
        args.tau = 0.5
        args.inner_update_step = 1
        # args.inner_batch_size = args.num_task_test
        learner = sustain.Learner(args, training_size)

    elif args.methods == 'vrbo':
        args.outer_update_lr = 1e-2
        args.inner_update_lr = 5e-3
        args.hessian_lr = 1e-2
        args.grad_normalized =  True
        args.grad_clip = False
        args.hessian_q = 3
        args.update_interval = 2
        args.spider_loops = 2
        # args.y_warm_start = 10
        args.gamma = 0.8
        args.beta = 0.9
        # args.alpha = 0.0
        args.tau = 0.5
        args.inner_update_step = 1

        args.inner_batch_size = 64
        learner = vrbo.Learner(args, training_size)

    elif args.methods == 'accbo':
        # args.outer_update_lr = 1e-1
        # args.inner_update_lr = 5e-2
        args.outer_update_lr = 5e-3
        args.inner_update_lr = 5e-3
        args.hessian_lr = 1e-1
        args.grad_normalized =  True
        args.grad_clip = False
        args.hessian_q = 3
        args.y_warm_start = 3
        args.gamma = 0.5
        args.beta = 0.9
        # args.alpha = 0.0
        args.tau = 0.5
        args.inner_update_step = 1
        # args.inner_batch_size = args.num_task_test
        learner = accbo.Learner(args, training_size)
    else:
        print('No such method, please change the method name!')


    global_step = 0
    auc_all_test = []
    loss_all_test = []
    auc_all_train = []
    loss_all_train = []
    # train_loader = DataLoader(train, shuffle=True,  batch_size=args.batch_size, collate_fn=collate_pad)
    # test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_pad)
    for epoch in range(args.epoch):
        print(f"[epoch/epochs]:{epoch}/{args.epoch}")
        train_loader = DataLoader(train, shuffle=True, batch_size=args.inner_batch_size, collate_fn=collate_pad)
        val_loader = DataLoader(val, shuffle=True, batch_size=args.batch_size, collate_fn=collate_pad)
        test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate_pad)
        auc, loss = learner(train_loader, val_loader, training=True, epoch=epoch)
        auc_all_train.append(auc)
        loss_all_train.append(loss)
        print('training Loss:', loss_all_train)
        print( 'training Auc:', auc_all_train)

        print("---------- Testing Mode -------------")

        auc, loss = learner.test(test_loader)
        auc_all_test.append(auc)
        loss_all_test.append(loss)

        print(f'{args.methods} Test loss:, {loss_all_test}')
        print(f'{args.methods} Test auc:, {auc_all_test}')
        global_step += 1

    date = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
    file_name = f'{args.methods}_outlr{args.outer_update_lr}_inlr{args.inner_update_lr}_seed{args.seed}_{date}'
    # acc_file_name = f'meta_learning_{args.methods}_{date}'
    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    save_path = 'logs/'
    # save_path = os.path.join(save_path, 'turning_gamma')
    total_time = (time.time() - st) / 3600
    files = open(os.path.join(save_path, file_name)+'.txt', 'w')
    files.write(str({'Exp configuration': str(args), 'AVG Train AUC': str(auc_all_train),
               'AVG Test AUC': str(auc_all_test), 'AVG Train LOSS': str(loss_all_train), 'AVG Test LOSS': str(loss_all_test),'time': total_time}))
    files.close()
    torch.save((auc_all_train, auc_all_test, loss_all_train, loss_all_test), os.path.join(save_path, file_name))
    print(args)
    print(f'time:{total_time} h')
if __name__ == "__main__":
    main()