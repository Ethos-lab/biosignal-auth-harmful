"""
Main script for training Deep-ECG
"""

import argparse
import os
import torch
import math
import numpy as np
import sys

from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim


from model import DeepECGModel, weights_init
from dataset import TrainDataset

from train_utils import EarlyStopper
import logging


def train(model, dataloader, optimizer, criterion, device, e, log_interval):
    model.train()

    epoch_loss = 0
    correct = 0
    for batch_idx, (x, y) in enumerate(dataloader):

        x, y = x.to(device), y.to(device)
        x = x.to(torch.float) # idk why

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()

        if batch_idx % log_interval == 0 and batch_idx > 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batch_idx * len(x), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

        #scheduler.step()

    epoch_acc = correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def test(model, dataloader, criterion, device):
    model.eval()
    preds = []
    trues = []
    epoch_loss = 0

    with torch.no_grad():
        for x, y in test_loader:

            trues.append(y.numpy())

            x, y = x.to(device), y.to(device)
            x = x.to(torch.float)
            out = model(x)
            loss = criterion(out, y)
            epoch_loss += loss.item()

            pred = out.argmax(1)
            preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        correct = np.equal(trues, preds).sum()
        epoch_acc = correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_interval', type=float, default=math.inf)
    parser.add_argument('--save', type=str, default='', help='if provided, save name')
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('--log', action='store_true', help='saves a log file with output')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    
    dataset = args.dataset
    epochs = args.epochs  # they did 500 epochs
    device = 'cuda:0'
    log_interval = args.log_interval

    # Hardcoding some training choices here for the strongest case for each 
    # Training until close to 100% test set accuracy for each dataset
    if args.dataset == 'ptbd':
        lr = 0.01
    elif args.dataset == 'bidmc': 
        lr = 0.001
        optim_type = 'sgd'
    elif args.dataset == 'capno': 
        lr = 0.001
        optim_type = 'sgd'
    elif args.dataset == 'dalia': 
        lr = 0.001
        optim_type = 'sgd'
        epochs = 120
    elif args.dataset == 'wesad': 
        lr = 0.001
        optim_type = 'sgd'
    elif args.dataset == 'hcitagging':
        lr = 0.001
        optim_type = 'sgd'
    else:
        lr = 0.01  # Default until we figure it out for that dataset
        optim_type = 'sgd'

    if args.log:
        # set up logging
        assert args.save, "If want to save log, should also provide save name with --save"
        savedir = os.path.join('saved_models', args.dataset)
        if not os.path.exists(savedir):  os.makedirs(savedir) # can rewrite
        logfn = os.path.join(savedir, f'{args.save}.log')
        logging.getLogger().addHandler(logging.FileHandler(logfn))

    weight_decay = 1e-4
    weight_decay = 1e-3
    batch_size = 512  # they did 500

    train_dataset = TrainDataset(dataset, 'train')  # NOTE: RESAMPLING TO 128 SO THAT TRAIN/SPOOF SAME SIZES
    num_classes = train_dataset.num_train_classes
    print('Loaded train dataset, size: ', len(train_dataset), ' Num classes: ', num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # TODO 

    test_dataset = TrainDataset(dataset, 'test')  # Thesea re the same patients but a different time: first 70% used to train, last 30% used for test/train cross validation during training 
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    # 256 is 10sec * fs, which for bidmc is 125. for bidmc. For capno is 608 for some reason, idk why but okay. 
    model = DeepECGModel(input_dim=train_dataset.sig_len, output_dim=num_classes)
    model = model.to(device)
    model.apply(weights_init)

    # Hardcoding some weird things
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
    else:
        raise NotImplementedError

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    criterion =  nn.CrossEntropyLoss()
    losses = []

    early_stopper = EarlyStopper(patience=5, min_delta=1e-4)  # NOTE changed from 1e-3
    schedule_stopper = EarlyStopper(patience=10, min_delta=1e-2) # just for scheduler.step


    # Train model 
    for e in range(epochs):
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, e, args.log_interval)

        test_loss, test_acc = test(model, test_loader, criterion, device)

        logging.info(f"Epoch {e}, Train loss: {train_loss:.3f}, Acc: {train_acc*100:.2f} Test loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}")

        if schedule_stopper.early_stop(test_loss): # if test loss not going anywhere, step scheduler
            scheduler.step()
            logging.info("stepping scheduler to: " + str(scheduler.get_last_lr()))

        if not args.no_early_stopping:
            if early_stopper.early_stop(test_loss):
                logging.info("Early stopping after: " + str(e) + " epochs")
                epochs = e
                break


    

    if args.save:
        logging.info("="*50)
        savedir = os.path.join('saved_models', args.dataset)
        if not os.path.exists(savedir):  os.makedirs(savedir)  # exist not okay
        savefn = os.path.join(savedir, f'{args.save}.pt')
        logging.info("Now saving to: " + savefn)
        model.eval()
        torch.save({
            'epochs': epochs,
            'fs': 128, 
            'losses': losses,
            'sig_len': train_dataset.sig_len,
            'num_train_classes': num_classes,
            'model_state_dict': model.state_dict()
            }, savefn)

