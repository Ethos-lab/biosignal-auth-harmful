#from cnn_model import getModel
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import math
from custom_datasets import datasets, processing
from torchmodels import BaseModel, init_weights
from segment_signals import segmentSignals
from train_utils import train_epoch, test_epoch, EarlyStopper
import os
import neurokit2 as nk2

class BaseClassifierDataset(Dataset):

    def __init__(self, name, split):
        super().__init__()
        # NOTE important, using all patients but (0.5, 0.625) for train, (0.625, 0.75) for crossval
        fraction_time = (0.5, 0.675) if split == 'train' else (0.675, 0.75) # 70/30 split

        self.DATA, self.PATIENT_IX = [], []

        fs = datasets.get_orig_fs(name, datatype='ecg', spoof=False)  # get orig fs for resampling to 128
        print("Adding ecg_clean")
    
        for pix, patient, sess, ecg in datasets.get_data(name, 'ecg', spoof=False, split='all', fraction_time=fraction_time):
            if name == 'hcitagging':
                ecg = nk2.ecg_clean(ecg[0], sampling_rate=fs)
                ecg = np.expand_dims(ecg, 0)
            ecg, r_peaks = processing.resample_get_rpeaks(ecg, fs, resample_to=128)  # needs an array of r_peaks
            segments, r_peaks = segmentSignals(ecg, r_peaks, person_id=pix, FS=128)
            segments = np.expand_dims(segments, 1)  # num x 1 x 256
            self.DATA.extend(segments)
            self.PATIENT_IX.extend([pix for i in range(len(segments))])
        self.DATA = np.stack(self.DATA)
        self.NUM_PATIENTS = len(set(self.PATIENT_IX))
        print('Loaded ' + split + ' data of size: ' + str(self.DATA.shape))

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, ix):
        return self.DATA[ix], self.PATIENT_IX[ix]

def train_base_classifier(dataset, epochs, device='cuda:0', save=False, log_interval=math.inf, no_early_stopping=False):

    train_dset, test_dset  = BaseClassifierDataset(dataset, 'train'), BaseClassifierDataset(dataset, 'test')
    train_loader, test_loader = DataLoader(train_dset, batch_size=64, shuffle=True), DataLoader(test_dset, batch_size=64)

    model = BaseModel(seq_len=train_dset[0][0].size, n_classes=train_dset.NUM_PATIENTS)
    model.to(device)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.001, eps=1e-5, verbose=True)
    early_stopper = EarlyStopper(patience=3, min_delta=1e-3)

    for e in range(epochs): 
        train_loss, train_acc = train_epoch(e, model, train_loader, optimizer, criterion, device, log_interval)
        print(f"Epoch {e}, Train loss: {train_loss:.3f}, Acc: {train_acc*100:.2f} ", end='')

        test_loss, test_acc = test_epoch(e, model, test_loader, criterion, device)  # TODO args
        print(f"Test loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}")

        if not no_early_stopping and  early_stopper.early_stop(test_loss):
            print("Early stopping after ", e, " epochs")
            epochs = e
            break

        scheduler.step(test_loss)


    if save:
        # DO SAVE
        print("="*50)
        savedir = os.path.join('saved_models', dataset)
        if not os.path.exists(savedir):  os.makedirs(savedir)
        savefn = os.path.join(savedir, f'{save}.pt')
        torch.save({
            'epochs': epochs,
            'fs': 128,
            'seq_len': train_dset[0][0].size,
            'num_train_classes': train_dset.NUM_PATIENTS,
            'model_state_dict': model.state_dict()
        }, savefn)
        print('Saved to: ', savefn)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=float, default=math.inf)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save', type=str, default='', help='if provided, save filename')
    parser.add_argument('--no_early_stopping', action='store_true', help='Otherwise it does')
    args = parser.parse_args() 
    train_base_classifier(args.dataset, args.epochs, 'cpu' if args.cpu else 'cuda:0', args.save, args.log_interval, args.no_early_stopping)
