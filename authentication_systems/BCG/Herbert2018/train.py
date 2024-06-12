from datasets import AllUsersDataset, SingleUserDataset

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from argparse import ArgumentParser
import os
import numpy as np

from model import Model, init_weights
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

    train_dset, test_dset = AllUsersDataset(args.dataset, split='train'), AllUsersDataset('bedbased', split='test')   # default num_heartbates=15
    eval_dset = AllUsersDataset(args.dataset, split='eval')
    num_classes = len(set(train_dset.PATIENT_IX))
    print("Num classes: ", num_classes)


    ALL_EVAL_EERS = []
    for u in set(train_dset.PATIENT_IX):

        train_loader = DataLoader(SingleUserDataset(train_dset, u, expansion_len=10), batch_size=128, shuffle=True)
        test_loader = DataLoader(SingleUserDataset(test_dset, u, expansion_len=10), batch_size=128, shuffle=True)

        device = 'cuda:1'
        model = Model(sample_len=train_dset.SAMPLE_LEN)
        model.apply(init_weights)
        model.to(device)
        criterion = nn.BCELoss()  
        lr = 0.0001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=25, min_delta=1e-5)


        for e in range(args.epochs):

            train_loss, train_acc, train_f1 = train_epoch(e, model, train_loader, optimizer, criterion, device)
            if args.verbose:  print(f"Epoch {e}, Train loss: {train_loss:.3f}, Acc: {train_acc*100:.2f}, F1: {train_f1:.2f}", end='')

            test_loss, test_acc, test_f1 = test_epoch(e, model, test_loader, criterion, device)
            if args.verbose:  print(f"Test loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}, F1: {test_f1:.2f}")


            if early_stopper.early_stop(test_loss):
                print("Early stopping after ", e)
                break

    
        # Now get eer and eer_thresh, potentially save
        eval_loader = DataLoader(SingleUserDataset(eval_dset, u, expansion_len=5), batch_size=64)
        eer, eer_thresh = eval_eer(model, eval_loader, device)
        ALL_EVAL_EERS.append(eer)

        print(f"User: {u}, Test ACC: {test_acc:.2f}, Test F1: {test_f1:.2f}, EER: {eer:.2f} at EER_THRESH: {eer_thresh:.3f}")

        if args.save:
            savedir = os.path.join('saved_models', args.dataset, args.save)
            if not os.path.exists(savedir):  os.makedirs(savedir)
            savefn = os.path.join(savedir, f'{u}.pt')
            torch.save({
                'EER': eer,
                'EER_THRESH': eer_thresh,
                'model_state_dict': model.state_dict(),
                }, savefn)

    print()
    print("Average EER across all users:", np.mean(ALL_EVAL_EERS))


if __name__ == "__main__":
    parser = ArgumentParser('Train Hebert 2018 BCG CNN')
    parser.add_argument('--dataset', type=str, default='bedbased')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    train(args)
