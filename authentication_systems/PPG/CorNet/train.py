from datasets import AllUsersDataset, SingleUserDataset

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from argparse import ArgumentParser
import os
import numpy as np

from model import Model
from custom_datasets import datasets, processing

import utils
from utils import train_epoch, test_epoch, eval_eer
import random


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(args):

    train_dset = AllUsersDataset(args.dataset, split='train', IGNORE_FRACTION_TIME=args.IGNORE_FRACTION_TIME)
    test_dset = AllUsersDataset(args.dataset, split='test', IGNORE_FRACTION_TIME=args.IGNORE_FRACTION_TIME)
    num_classes = len(set(train_dset.PATIENT_IX))
    print("Loaded datasets. Sizes: ", len(train_dset), len(test_dset))
    print("Num classes: ", num_classes)

    ALL_EVAL_EERS = []
    for u in set(train_dset.PATIENT_IX):
        
        single_user_train_dset = SingleUserDataset(train_dset, u, expansion_len=args.expansion_len)
        single_user_test_dset = SingleUserDataset(test_dset, u, expansion_len=args.expansion_len)

        train_loader = DataLoader(single_user_train_dset, batch_size=64, shuffle=True)
        test_loader = DataLoader(single_user_test_dset, batch_size=64, shuffle=False)

        if args.verbose:
            print("Size of single user sets: ", len(single_user_train_dset), len(single_user_test_dset))


        device = f'cuda:{args.gpu}'
        model = Model()
        model.to(device)
        criterion1 = nn.L1Loss()
        criterion2 = nn.BCELoss()
        lr = 0.0001  # TODO changed from 0.0001
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=10, min_delta=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

        for e in range(args.epochs):

            train_loss, train_acc = train_epoch(e, model, train_loader, optimizer, criterion1, criterion2, device, ALPHA=args.alpha)
            if args.verbose:  print(f"Epoch {e}, Train loss: {train_loss:.3f}, Acc: {train_acc*100:.2f} ", end='')

            test_loss, test_acc = test_epoch(e, model, test_loader, criterion1, criterion2, device, ALPHA=args.alpha)
            if args.verbose:  print(f"Test loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}")

            if not args.no_early_stopping and early_stopper.early_stop(test_loss):
                print("Early stopping at: ", e)
                break

            #scheduler.step()
        
        # Now get eer and eer_thresh, potentially save
        #eval_loader = DataLoader(SingleUserDataset(eval_dset, u, p=0.5), batch_size=64)
        eer, eer_thresh = eval_eer(model, test_loader, device)
        ALL_EVAL_EERS.append(eer)

        print(f"User: {u}, Test ACC: {test_acc:.2f}, EER: {eer:.2f} at EER_THRESH: {eer_thresh:.3f}")


        if args.save:
            savedir = os.path.join('saved_models', args.dataset, args.save)
            if not os.path.exists(savedir):  os.makedirs(savedir)
            savefn = os.path.join(savedir, f'{u}.pt')
            torch.save({
                'EER': eer,
                'EER_THRESH': eer_thresh,
                'model_state_dict': model.state_dict(),
                'alpha': args.alpha  # just so we know
                }, savefn)

    print()
    print("Average Test set EER across all users:", np.mean(ALL_EVAL_EERS))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)  # equal weight
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('--expansion_len', type=int, default=-1, help='Helps account for lots of patients...see singleuserdataset')
    parser.add_argument('--IGNORE_FRACTION_TIME', action='store_true')
    args = parser.parse_args()
    train(args)
