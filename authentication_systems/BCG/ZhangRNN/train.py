from datasets import TrainDataset

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




class EvalDataset(Dataset):
    def __init__(self, dset1, dset2, num_comps=500):
        """ Basically get random pairs of each, then there's a matching """
        self.dset1 = dset1
        self.dset2 = dset2
        self.num_comps=num_comps
        
        # for convenience:
        self.num_ixs = len(self.dset1)
        self.unique_users = list(set(self.dset1.PATIENT_IX))

    def __len__(self):
        return self.num_comps

    def __getitem__(self, ix):
        # Get a pair of data from dset1, dset2, and whether they're a match. Try to balance so use prob = 50

        rand_ix1 = random.randint(0, self.num_ixs-1)
        data1, user1 = self.dset1[rand_ix1]

        same = bool(random.getrandbits(1))
        if same:
            subset = [i for i, u in enumerate(self.dset2.PATIENT_IX) if u == user1 and i < len(self.dset2)]
        else:
            subset = [i for i, u in enumerate(self.dset2.PATIENT_IX) if u != user1 and i < len(self.dset2)]
        rand_ix2 = random.choice(subset)
        if rand_ix2 >= len(self.dset2):  import pdb; pdb.set_trace()
        data2, _ = self.dset2[rand_ix2]
        return data1, data2, float(same)

    

def train(args):

    T = args.T
    train_dset, test_dset = TrainDataset(split='train', T=T, both_bcg_ecg=args.both), TrainDataset(split='test', T=T, both_bcg_ecg=args.both)   # default num_heartbates=15
    train_loader, test_loader = DataLoader(train_dset, batch_size=128, shuffle=True), DataLoader(test_dset, batch_size=128)
    num_classes = len(set(train_dset.PATIENT_IX))
    print("Num classes: ", num_classes)

    device = 'cuda:0'
    model = Model(T=T, num_classes=num_classes, input_size=train_dset.SAMPLE_LEN)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.01, eps=1e-7, verbose=True)


    for e in range(args.epochs):

        train_loss, train_acc = train_epoch(e, model, train_loader, optimizer, criterion, device)
        print(f"Epoch {e}, Train loss: {train_loss:.3f}, Acc: {train_acc*100:.2f} ", end='')

        test_loss, test_acc = test_epoch(e, model, test_loader, criterion, device)
        print(f"Test loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}")

    train_pairs = EvalDataset(dset1=train_dset, dset2=train_dset, num_comps=1000)
    loader = DataLoader(train_pairs, batch_size=128, shuffle=False)
    eer, eer_thresh = eval_eer(model, loader, device)
    acc_at_eer_thresh = utils.predict(model, loader, device, eer_thresh)
    print("="*50)
    print("Metrics on train train set")
    print(f"EER: {eer}, EER_THRESH: {eer_thresh}")
    print("Acc at eer thresh: ", acc_at_eer_thresh)

    # and now eval on test and eval dataset  using matching/l2 distance, pairs etc 
    eval_pairs = EvalDataset(dset1=train_dset, dset2=test_dset, num_comps=500)
    eval_loader = DataLoader(eval_pairs, batch_size=128, shuffle=False)
    eer, eer_thresh = eval_eer(model, eval_loader, device)
    print("="*50)
    print("Metrics on training test set: ")
    print(f"EER: {eer}, EER_THRESH: {eer_thresh}")

    if args.save:
        print("="*50)
        savedir = 'saved_models'
        if not os.path.exists(savedir):  os.makedirs(savedir)
        savefn = os.path.join(savedir, f'{args.save}.pt')
        torch.save({
            'epochs': args.epochs,
            'T': args.T,
            'num_classes': num_classes,
            'input_size': train_dset.SAMPLE_LEN,
            'EER': eer,
            'EER_THRESH': eer_thresh,
            'model_state_dict': model.state_dict()
        }, savefn)
        print('Saved to: ', savefn)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--save', type=str, default='', help='Save filename, default not to save')
    parser.add_argument('--T', type=int, default=15, help='Num consecutive hbs')
    parser.add_argument('--epochs', type=int, default=15, help='eopchs')
    parser.add_argument('--both', action='store_true', help='Add in ecg')
    args = parser.parse_args()
    train(args)
