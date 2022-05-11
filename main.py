#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import scipy.io
import numpy as np

import torch
#import dataset
from vae import MultiVAE
from trainer import Trainer
import utils
import tqdm
import pandas as pd
import datasets


configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1e-4,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.1, # "lr_policy: step"
        step_size=200000, # "lr_policy: step" e-6
        interval_validate=1000,
    ),
}

def main():
    parser = argparse.ArgumentParser("Variational autoencoders for collaborative filtering")
    parser.add_argument('cmd', type=str,  choices=['train','test'], help='train')
    parser.add_argument('--arch_type', type=str, default='MultiVAE', help='architecture', choices=['MultiVAE', 'MultiDAE'])
    parser.add_argument('--dataset_name', type=str, default='lfm2b', help='camera model type', choices=['ml-20m', 'netflix','lfm2b','ml1m'])
    parser.add_argument('--processed_dir', type=str, default='../data/lfm2b/', help='dataset directory')
    parser.add_argument('--n_items', type=int, default=1, help='n items')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='checkpoints directory')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='checkpoint save frequency')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--start_step', dest='start_step', type=int, default=0, help='start step')
    parser.add_argument('--total_steps', dest='total_steps', type=int, default=int(3e5), help='Total number of steps')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--train_batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=100, help='batch size in validation')
    parser.add_argument('--test_batch_size', type=int, default=100, help='batch size in test')
    parser.add_argument('--print_freq', type=int, default=1, help='log print frequency')
    parser.add_argument('--upper_train', type=int, default=-1, help='max of train images(for debug)')
    parser.add_argument('--upper_valid', type=int, default=-1, help='max of valid images(for debug)')
    parser.add_argument('--upper_test', type=int, default=-1, help='max of test images(for debug)')
    parser.add_argument('--total_anneal_steps', type=int, default=2000, help='the total number of gradient updates for annealing')
    parser.add_argument('--beta', type=float, default=1.0, help='largest annealing parameter')
    parser.add_argument('--gamma', type=float, default=5.0, help='largest annealing parameter')
    parser.add_argument('--tau', type=float, default=0.5, help='largest annealing parameter')
    parser.add_argument('--dropout_p', dest='dropout_p', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--target', type=str, default="country", help='Activate Bias Method')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    args = parser.parse_args()


    #if args.cmd == 'train':
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    cfg = configurations[args.config]

    #print("\n".join(args))
    #print("-----------------------------------")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    if cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    if not os.path.isdir("./checkpoint"): os.mkdir("./checkpoint")
    if not os.path.isdir("./results"): os.mkdir("./results")

    torch.manual_seed(98765)
    if cuda:
        torch.cuda.manual_seed(98765)

    # # 1. data loader
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    root = args.processed_dir

    item_mapper = pd.read_csv(root+"item_mapper.csv", header=0)
    user_mapper = pd.read_csv(root+"user_mapper.csv", header=0)

    #################################

    #target = "country"
    if args.target == "country" and args.dataset_name == 'lfm2b':
        labels = pd.DataFrame(user_mapper["country"].value_counts())
        labels.columns = ['counts']
        labels["country"] = labels.index

        labels_others = labels.loc[labels.counts < 500]
        labels = labels.replace(labels_others.country.values, "OTHERS").filter(["country"]).drop_duplicates()

        labels = labels.reset_index(drop=True)

        #labels = df.filter(["country"]).drop_duplicates().reset_index(drop=True)
        labels["country_id"] = labels.index

        user_mapper["country"] = user_mapper.country.replace(labels_others.country.values, "OTHERS")
        user_mapper = user_mapper.merge(labels, on="country", how="inner")
        args.target = "country_id"

    elif args.target == "country":
        labels = user_mapper[args.target].unique()
    elif args.target == "gender":
        labels = user_mapper[args.target].unique()

    #################################

    unique_sid = item_mapper.new_movieId.unique()
    args.n_items = len(unique_sid)

    DS = datasets.LFM1bDataset
    dt = DS(root, item_mapper, user_mapper, args.target, split='train', upper=args.upper_train, conditioned_on=args.conditioned_on)
    train_loader = torch.utils.data.DataLoader(dt, batch_size=args.train_batch_size, shuffle=False, **kwargs)

    dt = DS(root, item_mapper, user_mapper, args.target, split='valid', upper=args.upper_valid, conditioned_on=args.conditioned_on)
    valid_loader = torch.utils.data.DataLoader(dt, batch_size=args.valid_batch_size, shuffle=False, **kwargs)

    dt = DS(root, item_mapper, user_mapper, args.target, split='test', upper=args.upper_test, conditioned_on=args.conditioned_on)
    test_loader = torch.utils.data.DataLoader(dt, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # 2. model
    n_conditioned = 0
    if args.cmd == 'train':
        model = MultiVAE(dropout_p=args.dropout_p, weight_decay=0.0, cuda2=cuda,
                             q_dims=[args.n_items, 2000, 200], p_dims=[200, 2000, args.n_items], 
                             n_conditioned=n_conditioned, n_classes=len(labels))
        # 3. optimizer
        optim = torch.optim.Adam(
            [
                {'params': list(utils.get_parameters(model, bias=False)), 'weight_decay': 0.0},
                {'params': list(utils.get_parameters(model, bias=True)), 'weight_decay': 0.0},
            ],
            lr=cfg['lr'],
        )

    elif args.cmd == 'test':
        model = MultiVAE(dropout_p=args.dropout_p, weight_decay=0.0, cuda2=cuda,
                             q_dims=[args.n_items, 2000, 200], p_dims=[200, 2000, args.n_items], 
                             n_conditioned=n_conditioned, n_classes=len(labels))
        # 3. optimizer
        optim = torch.optim.Adam(
            [
                {'params': list(utils.get_parameters(model, bias=False)), 'weight_decay': 0.0},
                {'params': list(utils.get_parameters(model, bias=True)), 'weight_decay': 0.0},
            ],
            lr=cfg['lr'],
        )
        if cuda: model = model.cuda()

        checkpoint = torch.load('checkpoint/{}_vae_{}_{}_{}_{}.pth'.format(args.dataset_name, args.target, args.beta, args.gamma, args.tau))
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    print(model)

    start_epoch = 0
    start_step = 0

    if cuda:
        model = model.cuda()


    # lr_policy: step
    last_epoch = -1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[50, 75], gamma=cfg['gamma'], last_epoch=last_epoch)

    if args.cmd == 'train':
        trainer = Trainer(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            optim=optim,
            gamma = args.gamma,
            tau = args.tau,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            start_step=start_step,
            total_steps=args.total_steps,
            interval_validate=args.valid_freq,
            checkpoint_dir=args.checkpoint_dir,
            print_freq=args.print_freq,
            checkpoint_freq=args.checkpoint_freq,
            total_anneal_steps=args.total_anneal_steps,
            beta=args.beta,
            item_mapper = item_mapper,
            user_mapper = user_mapper,
            dataset_name = args.dataset_name,
            alpha = args.alpha,
            base_dir = root,
            target=args.target
        )
        trainer.train()
    else:

        trainer = Trainer(
            cmd=args.cmd,
            cuda=cuda,
            model=model,
            optim=optim,
            gamma = args.gamma,
            tau = args.tau,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            start_step=start_step,
            total_steps=args.total_steps,
            interval_validate=args.valid_freq,
            checkpoint_dir=args.checkpoint_dir,
            print_freq=args.print_freq,
            checkpoint_freq=args.checkpoint_freq,
            total_anneal_steps=args.total_anneal_steps,
            beta=args.beta,
            item_mapper = item_mapper,
            user_mapper = user_mapper,
            dataset_name = args.dataset_name,
            base_dir = root,
            target=args.target
            )
        trainer.test()


if __name__ == '__main__':
    main()
