#!/usr/bin/env python

import collections
import os

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import time
import utils
import glob
import tqdm
from scipy import sparse
import pandas as pd

class LFM1bDataset(data.Dataset):

    def __init__(self, root, item_mapper, user_mapper, target, fold_in=True, split='train', conditioned_on=None, upper=-1):

        super(LFM1bDataset, self).__init__()
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        #assert len(img_files) > 0
        #self.img_files = img_files
        assert os.path.exists(root), "root: {} not found.".format(root)

        assert split in ['test', 'inference', 'train', 'valid']
        #fname = os.path.join(root, self.img_files)

        #self.train_data, self.vad_data_tr, self.vad_data_te, self.test_data_tr, self.test_data_te = utils.load_weights_pkl(fname)

        out_data_dir = root
        self.target = target

        self.user_mapper = user_mapper

        if target == "gender":
            self.user_mapper = self.user_mapper.replace({'m': 0, 'f': 1, 'n':2})

        unique_sid = item_mapper.new_movieId.unique()
        #unique_sid = list()
        #with open(os.path.join(out_data_dir, 'unique_sid.txt'), 'r') as f:
        #    for line in f:
        #        unique_sid.append(line.strip())

        n_items = len(unique_sid)

        self.train_data, tr_start_idx, tr_end_idx = utils.load_train_data(os.path.join(out_data_dir, 'train.csv'), n_items)
        #self.train_data = utils.load_train_data(os.path.join(out_data_dir, 'train.csv'), n_items)
        self.vad_data_tr, self.vad_data_te, vad_start_idx, vad_end_idx = utils.load_tr_te_data(os.path.join(out_data_dir, 'validation_tr.csv'), os.path.join(out_data_dir, 'validation_te.csv'), n_items)
        self.test_data_tr, self.test_data_te, te_start_idx, te_end_idx = utils.load_tr_te_data(os.path.join(out_data_dir, 'test_tr.csv'), os.path.join(out_data_dir, 'test_te.csv'), n_items)

        assert self.train_data.shape[1] == self.vad_data_tr.shape[1]
        assert self.vad_data_tr.shape == self.vad_data_te.shape
        assert self.test_data_tr.shape == self.test_data_te.shape

        self.split = split
        self.fold_in = fold_in

        self.te_idx = np.arange(te_start_idx, te_end_idx+1, 1)
        self.vad_idx = np.arange(vad_start_idx, vad_end_idx+1, 1)
        self.tr_idx = np.arange(tr_start_idx, tr_end_idx+1, 1)
        print("USERS TEST FROM {} TO {}".format(te_start_idx,te_end_idx))

        if self.split == 'train':
            self.n_users = self.train_data.shape[0]
        elif self.split == 'valid':
            self.n_users = self.vad_data_tr.shape[0]
        elif self.split == 'test':
            self.n_users = self.test_data_tr.shape[0]
        else:
            raise NotImplementedError

        self.n_users = self.n_users if upper <= 0 else min(self.n_users, upper)

    def __len__(self):
        return self.n_users

    def __getitem__(self, index):
        prof = np.zeros(1)
        if self.split == 'train':
            data_tr, data_te = self.train_data[index], np.zeros(1)
            idx_user = self.tr_idx[index]
        elif self.split == 'valid':
            data_tr, data_te = self.vad_data_tr[index], self.vad_data_te[index]
            idx_user = self.vad_idx[index]
        elif self.split == 'test':
            data_tr, data_te = self.test_data_tr[index], self.test_data_te[index]
            idx_user = self.te_idx[index]

        if sparse.isspmatrix(data_tr):
            data_tr = data_tr.toarray()
        data_tr = data_tr.astype('float32')
        data_tr = data_tr[0]

        if sparse.isspmatrix(data_te):
            data_te = data_te.toarray()
        data_te = data_te.astype('float32')
        data_te = data_te[0]

        sensitive = self.user_mapper.loc[self.user_mapper.new_userId == idx_user][self.target].values[0]

        return data_tr, data_te, prof, idx_user, sensitive
