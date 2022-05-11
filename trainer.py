# -*- coding: utf-8 -*-

import datetime
import math
import os
import time

import numpy as np
import scipy.io
from scipy import sparse
import torch
from torch.autograd import Variable
import torch.nn.functional as f

import utils
from utils import AverageMeter
import tqdm
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class Trainer(object):

    def __init__(self, cmd, cuda, model, optim=None,
                 train_loader=None, valid_loader=None, test_loader=None, log_file=None,
                 interval_validate=1, lr_scheduler=None, dataset_name = None, gamma=0.0, tau=0.0,
                 start_step=0, total_steps=1e5, beta=0.05, start_epoch=0, bias=False, target=None,
                 total_anneal_steps=200000, beta=0.1, do_normalize=True, item_mapper = None, user_mapper = None,
                 checkpoint_dir=None, result_dir=None, print_freq=1, result_save_freq=1, checkpoint_freq=1, base_dir=None):

        self.cmd = cmd
        self.cuda = cuda
        self.model = model
        self.item_mapper = item_mapper
        self.user_mapper = user_mapper
        self.dataset_name = dataset_name
        self.bias = bias
        self.base_dir = base_dir

        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now()

        if self.cmd == 'train':
            self.interval_validate = interval_validate

        self.start_step = start_step
        self.step = start_step
        self.total_steps = total_steps
        self.epoch = start_epoch

        self.do_normalize = do_normalize
        self.print_freq = print_freq
        self.checkpoint_freq = checkpoint_freq

        self.checkpoint_dir = checkpoint_dir

        self.total_anneal_steps = total_anneal_steps
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.ndcg, self.recall, self.ash, self.amt, self.alt, self.ent, self.demo = [], [], [], [], [], [], []
        self.loss, self.kl, self.posb, self.popb = [],[],[],[]
        self.neg, self.kl, self.ubias = [], [], []
        
        self.target = target
        self.criterion = torch.nn.CrossEntropyLoss()
     

    def validate(self, cmd="valid", k=100):
        assert cmd in ['valid', 'test']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.eval()

        end = time.time()

        n10_list, n100_list, r10_list, r100_list = [], [], [], []
        embs_list = []
        att_round, rel_round, cnt_round, pcount_round, udx_list = [], [], [], [], []
        result = []
        eval_loss = 0.0
        eval_neg = 0.0
        eval_kl = 0.0
        eval_ubias = 0.0

        loader_ = self.valid_loader if cmd == 'valid' else self.test_loader

        step_counter = 0
        for batch_idx, (data_tr, data_te, prof, uindex, sens) in tqdm.tqdm(enumerate(loader_), total=len(loader_),
                                   desc='{} check epoch={}, len={}'.format('Valid' if cmd == 'valid' else 'Test',
                                                               self.epoch, len(loader_)), ncols=80, leave=False):
            step_counter = step_counter + 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad():
                logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)
            
                log_softmax_var = f.log_softmax(logits, dim=1)
                neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))
                eval_neg += neg_ll.item()
                eval_kl += KL.item()
            
                user_bias = utils.calc_user_bias(torch.sum(log_softmax_var * data_tr, dim=1), sens) 
                eval_ubias += user_bias.item()

                # classification accuracy            
                y_hat = self.model.classify(sampled_z)
                if self.cuda:
                    class_loss = self.criterion(y_hat, torch.flatten(Variable(sens.type(torch.LongTensor))).cuda())
                else:
                    class_loss = self.criterion(y_hat, torch.flatten(Variable(sens.type(torch.LongTensor))))

                eval_loss += class_loss.item()

                pred_val = logits.cpu().detach().numpy()
                pred_val[data_tr.cpu().detach().numpy().nonzero()] = -np.inf

                data_te_csr = sparse.csr_matrix(data_te.numpy())
                n10_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te_csr, k=10))
                n100_list.append(utils.NDCG_binary_at_k_batch(pred_val, data_te_csr, k=100))
                r10_list.append(utils.Recall_at_k_batch(pred_val, data_te_csr, k=10))
                r100_list.append(utils.Recall_at_k_batch(pred_val, data_te_csr, k=100))


                if cmd == 'test':
                    for user in np.arange(data_te.numpy().shape[0]):
                        dict_out={}
                        preds = pred_val[user,:]

                        dict_out['num_missing_terms'] = len(np.array(data_te.numpy()[user,:]).nonzero()[0])
                        dict_out['missing_terms'] = " ".join([str(x) for x in list(np.array(data_te.numpy()[user,:]).nonzero()[0])])
                        dict_out['num_terms'] = len(np.array(data_te.numpy()[user,:]).nonzero()[0]) + len(np.array(data_tr.cpu().detach().numpy()[user,:]).nonzero()[0])
                        dict_out['recommended_terms'] = " ".join([str(x) for x in list(np.argsort(-preds)[:k])])
                        dict_out['new_userId'] = int(uindex[user].cpu().detach().numpy())
                        dict_out['scores'] = " ".join([str(x) for x in list(np.sort(self.softmax(preds))[::-1][:k])])
                        result.append(dict_out)  

        avg_loss =eval_loss/len(loader_)
        avg_neg =eval_neg/len(loader_)
        avg_kl =eval_kl/len(loader_)
        avg_ubias =eval_ubias/len(loader_)

        metrics = []
        if cmd == 'valid':
        
            n10_list = np.concatenate(n10_list, axis=0)
            n100_list = np.concatenate(n100_list, axis=0)
            r10_list = np.concatenate(r10_list, axis=0)
            r100_list = np.concatenate(r100_list, axis=0)

            self.ndcg.append(np.mean(n100_list))
            self.recall.append(np.mean(r100_list))
            self.loss.append(avg_loss)
            self.neg.append(avg_neg)
            self.kl.append(avg_kl)
            self.ubias.append(avg_ubias)

            np.save('results/'+self.dataset_name+'_ndcg_{}_{}_{}_{}.npy'.format(self.target, self.beta, self.gamma, self.tau), self.ndcg)
            np.save('results/'+self.dataset_name+'_recall_{}_{}_{}_{}.npy'.format(self.target, self.beta, self.gamma, self.tau), self.recall)
            np.save('results/'+self.dataset_name+'_loss_{}_{}_{}_{}.npy'.format(self.target, self.beta, self.gamma, self.tau), self.loss)
            np.save('results/'+self.dataset_name+'_neg_{}_{}_{}_{}.npy'.format(self.target, self.beta, self.gamma, self.tau), self.neg)
            np.save('results/'+self.dataset_name+'_kl_{}_{}_{}_{}.npy'.format(self.target, self.beta, self.gamma, self.tau), self.kl)
            np.save('results/'+self.dataset_name+'_ubias_{}_{}_{}_{}.npy'.format(self.target, self.beta, self.gamma, self.tau), self.ubias)

            # SAVE MODEL
            torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict()
            }, self.checkpoint_dir+self.dataset_name+'_vae_{}_{}_{}_{}.pth'.format(self.target, self.beta, self.gamma, self.tau))
            #with open(self.checkpoint_dir+self.dataset_name+'_vae_'+str(self.bias)+'_'+str(self.alpha)+'.pt', 'wb') as model_file: torch.save(self.model, model_file)
            #torch.save({'state_dict': self.model.state_dict()}, self.checkpoint_dir+'vae')

            metrics.append("NDCG@10,{:.5f},{:.5f}".format(np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))
            metrics.append("NDCG@100,{:.5f},{:.5f}".format(np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
            metrics.append("Recall@10,{:.5f},{:.5f}".format(np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
            metrics.append("Recall@100,{:.5f},{:.5f}".format(np.mean(r100_list), np.std(r100_list) / np.sqrt(len(r100_list))))
            print('\n' + ",\n".join(metrics))

        else:

            final_results = pd.DataFrame(result)
            final_results = final_results.merge(self.user_mapper[['new_userId','gender','country','age']], on='new_userId',how='inner')
            final_results.to_csv("results/{}_final_results_{}_{}_{}_{}.csv".format(self.dataset_name, self.target, self.beta, self.gamma, self.tau), index=False)

        self.model.train()


    def train_epoch(self):
        cmd = "train"
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        self.model.train()

        end = time.time()
        for batch_idx, (data_tr, data_te, prof, uidx, sens) in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train check epoch={}, len={}'.format(self.epoch, len(self.train_loader)), ncols=80, leave=False):
            self.step += 1

            if self.cuda:
                data_tr = data_tr.cuda()
                prof = prof.cuda()
            data_tr = Variable(data_tr)
            prof = Variable(prof)
            data_time.update(time.time() - end)
            end = time.time()

            logits, KL, mu_q, std_q, epsilon, sampled_z = self.model.forward(data_tr, prof)

            log_softmax_var = f.log_softmax(logits, dim=1)
            neg_ll = - torch.mean(torch.sum(log_softmax_var * data_tr, dim=1))

            l2_reg = self.model.get_l2_reg()

            if self.total_anneal_steps > 0:
                self.anneal = min(self.beta, 1. * self.step / self.total_anneal_steps)
            else:
                self.anneal = self.beta

            ## CLASSIFICATION ACCURACY
            y_hat = self.model.classify(sampled_z)
            if self.cuda:
                class_loss = self.criterion(y_hat, torch.flatten(Variable(sens.type(torch.LongTensor))).cuda())
            else:
                class_loss = self.criterion(y_hat, torch.flatten(Variable(sens.type(torch.LongTensor))))

            # USER BIAS
            user_bias = utils.calc_user_bias(torch.sum(log_softmax_var * data_tr, dim=1), sens) 

            loss = neg_ll + self.anneal * KL + l2_reg  - self.gamma * class_loss + self.tau * user_bias 

            # backprop
            self.model.zero_grad()
            loss.backward()
            self.optim.step()


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train(self):

        max_epoch = 100
        for epoch in tqdm.trange(0, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            self.lr_scheduler.step()
            self.validate(cmd='valid')
            #self.validate(cmd='test')

    
    def test(self):
        self.validate(cmd='test')
