import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import math
from custom_datasets import datasets, processing
from torchmodels import BaseModel, SiameseModel, init_weights
from segment_signals import segmentSignals
from train_utils import train_siamese_epoch, test_siamese_epoch, eval_eer, EarlyStopper
import random
import os
import neurokit2 as nk2

class SiameseDataset(Dataset):

    def __init__(self, name, split, base_model, negative_prob=0.5):
        super().__init__()

        self.NEGATIVE_PROB = negative_prob

        # NOTE important, using all patients but (0.5, 0.625) for train, (0.625, 0.75) for crossval
        if split == 'train':
            fraction_time = (0.5, 0.675)
        elif split == 'test':
            fraction_time = (0.675, 0.75)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        
        self.DATA, self.PATIENT_IX = [], []

        fs = datasets.get_orig_fs(name, datatype='ecg', spoof=False)  # get orig fs for resampling to 128
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
        self.PATIENT_IX = np.array(self.PATIENT_IX)

        self.DATA = self.get_features(base_model, self.DATA)
        print('Loaded ' + split + ' data of size: ' + str(self.DATA.shape))

    def get_features(self, model, data):
        device = next(model.parameters()).device
        all_out = []
        batch_start, batch_end, batch_size = 0, 256, 256
        with torch.no_grad():
            while batch_end < len(data):
                batch = torch.FloatTensor(data[batch_start:batch_end,:,:])
                batch = torch.FloatTensor(batch).to(device)
                out = model.get_embedding(batch)
                all_out.append(out.cpu().numpy())
                batch_start += batch_size
                batch_end += batch_size
            if batch_start < len(data):
                batch = torch.FloatTensor(data[batch_start:,:,:])
                batch = torch.FloatTensor(batch).to(device)
                out = model.get_embedding(batch)
                all_out.append(out.cpu().numpy())
        all_out = np.concatenate(all_out)
        return all_out

    def __len__(self):
        return len(self.DATA)*20 # TODO approx what's in the paper but let's see

    def __getitem__(self, ix):
        ix1 = ix % len(self.PATIENT_IX) # because it will wrap around 20 times
        sub1 = self.PATIENT_IX[ix1]

        is_same = random.random() > self.NEGATIVE_PROB

        if is_same:
            other_subset = np.where(self.PATIENT_IX == sub1)[0]
        else:
            other_subset = np.where(self.PATIENT_IX != sub1)[0]

        ix2 = random.choice(other_subset)

        sig1, sig2 = self.DATA[ix1], self.DATA[ix2]
        return sig1, sig2, is_same

def train_siamese(dataset, base_model_fn, epochs, device='cuda:0', save=False, log_interval=math.inf, no_early_stopping=False):

    base_model_dict = torch.load(base_model_fn)
    seq_len, n_classes = base_model_dict['seq_len'], base_model_dict['num_train_classes']
    base_model = BaseModel(seq_len=seq_len, n_classes=n_classes)
    base_model.load_state_dict(base_model_dict['model_state_dict'])
    base_model.to(device)
    base_model.eval()

    train_dset = SiameseDataset(dataset, 'train', base_model)
    train_loader = DataLoader(train_dset, batch_size=64, shuffle=True)
    test_dset = SiameseDataset(dataset, 'test', base_model)
    test_loader = DataLoader(test_dset, batch_size=64)
    eval_dset = SiameseDataset(dataset, 'eval', base_model)
    eval_loader = DataLoader(eval_dset, batch_size=64, shuffle=True)

    # Load saved base classifier
    model = SiameseModel()
    # TODO torch.load
    model.apply(init_weights)
    model.to(device)
    print("Training on device: ", torch.cuda.get_device_name(device))

    lr = 0.01 # TODO BAH Wasnt here before
    if dataset == 'hcitagging':
        lr = 0.0001 # Try this
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.001, eps=1e-5, verbose=True)
    early_stopper = EarlyStopper(patience=3, min_delta=1e-3)

    for e in range(epochs): 
        train_loss = train_siamese_epoch(e, model, train_loader, optimizer, criterion, device, log_interval)
        print(f"Epoch {e}, Train loss: {train_loss:.3f} ", end='')

        test_loss = test_siamese_epoch(e, model, test_loader, criterion, device)  # TODO args
        eer, eer_thresh = eval_eer(model, test_loader, device)
        print(f"Test loss: {test_loss:.3f} and eer: {eer:.3f} at eer_thresh: {eer_thresh:.3f}")

        if not no_early_stopping and early_stopper.early_stop(test_loss):
            print("Early stopping after: ", e, " epochs")
            epochs = e
            break

        scheduler.step(test_loss)

    print()
    print("="*20)
    eer, eer_thresh = eval_eer(model, eval_loader, device)
    print("Validation EER: ", eer, " at threshold: ", eer_thresh)

    if save:
        # DO SAVE
        print("="*50)
        savedir = os.path.join('saved_models', dataset)
        if not os.path.exists(savedir):  os.makedirs(savedir)
        savefn = os.path.join(savedir, f'{save}.pt')
        torch.save({
            'epochs': epochs,
            'EER': eer,
            'EER_THRESH': eer_thresh,
            'model_state_dict': model.state_dict()
        }, savefn)
        print("Saved to: ", savefn)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--saved_model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log_interval', type=float, default=math.inf)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save', type=str, default='', help='if provided, save filename')
    parser.add_argument('--no_early_stopping', action='store_true')
    args = parser.parse_args() 
    assert args.saved_model
    train_siamese(args.dataset, args.saved_model, args.epochs, 'cpu' if args.cpu else 'cuda:0', args.save, args.log_interval, args.no_early_stopping)
