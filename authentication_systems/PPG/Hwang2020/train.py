"""
For each patient, train a binary classifier. 
"""
from utils import train_epoch, test_epoch, eval_eer
from datasets import MulticlassDataset, BinaryDataset


import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import  DataLoader
from argparse import ArgumentParser
from custom_datasets import datasets, processing
import os
import random
import models

import signal_processing as sp
from sklearn.metrics import roc_curve



def train(args):

    # "for the distacne matching approach, we get the average of the training data for a given user and use that as the template."
    print("CHANNELS: ", args.channels)

    train_dset = MulticlassDataset(args.dataset, 'train', fs=args.fs, channels=args.channels, augmentation=args.augmentation)
    test_dset = MulticlassDataset(args.dataset, 'test', fs=args.fs, channels=args.channels, augmentation=False)
    num_users = train_dset.num_users

    device = 'cuda:'+str(args.gpu)

    criterion = nn.BCELoss()
    lr = 1e-4


    ALL_EVAL_EERS = []
    for user in set(train_dset.users):

        model = models.get_model(train_dset.num_channels, 1, lstm=args.lstm, binary=True)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)  

        user_train_dset = BinaryDataset(train_dset, user, neg_multiplier=2) 
        user_test_dset = BinaryDataset(test_dset, user, neg_multiplier=2) 

        train_loader = DataLoader(user_train_dset, batch_size=128, shuffle=True)
        test_loader = DataLoader(user_test_dset, batch_size=128)

        for e in range(args.epochs):

            train_loss, train_acc, train_metric = train_epoch(e, model, train_loader, optimizer, criterion, device, binary=True)
            if args.verbose:  print(f"Epoch {e}, Train loss: {train_loss:.3f}, Acc: {train_acc*100:.2f}, Metric: {train_metric:.2f}", end='  ')

            test_loss, test_acc, test_metric = test_epoch(e, model, test_loader, criterion, device, binary=True)
            if args.verbose:  print(f"Test loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}, M: {test_metric:.2f}")



        # Now test EER and save threshold
        eval_dset = MulticlassDataset(args.dataset, 'eval', fs=args.fs, channels=args.channels, augmentation=False)
        user_eval_dset = BinaryDataset(eval_dset, user, neg_multiplier=2)
        eval_loader = DataLoader(user_eval_dset, batch_size=128, shuffle=False)
        eer, eer_thresh, acc = eval_eer(model, eval_loader, device, binary=True) # this is the first problem, dont need it on the entir thing
        ALL_EVAL_EERS.append(eer)
         
        print(f"User: {user}, Test ACC: {test_acc:.2f}, Test M: {test_metric:.2f}, EER: {eer:.2f} at EER_THRESH: {eer_thresh:.3f} ACC@EER: {acc:.2f}")

        # and now save the model for this user
        if args.save:
            savedir = os.path.join('saved_models', args.dataset, args.save)
            if not os.path.exists(savedir):  os.makedirs(savedir)
            savefn = os.path.join(savedir, f'{user}.pt')
            torch.save({
                'EER': eer,
                'EER_THRESH': eer_thresh,
                'model_state_dict': model.state_dict(),
                'num_classes': 1,
                'channels': eval_dset.channels,
                'augmentation': args.augmentation,
                'num_channels': train_dset.num_channels,
                'lstm': args.lstm,
                }, savefn)
    

    print()
    print("Average EER across all users:", np.mean(ALL_EVAL_EERS))


if __name__ == '__main__':
    parser = ArgumentParser('Hwang 2020/2021 Wavelet PPG')
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--channels', type=str, nargs="+", default=['DTW', 'ZT'])
    parser.add_argument('--fs', type=int, default=128, help='fs to resample to')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--save', type=str, default='', help='If give, saves with this name')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    train(args)
