# import models  # importing dynamically in get_models
import importlib
from collections import defaultdict
from scipy import stats
from frechetdist import frdist
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
import math
#import biosppy
import neurokit2 as nk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# Makes checkpoint directories
def make_dirs(dataset, exp_name, force=False):
    ' Makes folders at ./ckpts/dataset/exp_name/.... Asks to rewrite, unless forced to rewrite '
    try:
        os.makedirs(f"ckpts/{dataset}/{exp_name}/weights")
        os.makedirs(f"ckpts/{dataset}/{exp_name}/images/train")
        os.makedirs(f"ckpts/{dataset}/{exp_name}/images/test")
        os.makedirs(f"ckpts/{dataset}/{exp_name}/images_best/train")
        os.makedirs(f"ckpts/{dataset}/{exp_name}/images_best/test")

    except OSError:
        if not force:
            answer = input(f"Folder '{exp_name}' already exists. Do you want to overwrite it? [yes(y) or no]")
        if force or  answer.lower() in ["y", "yes"]:
            os.makedirs(f"ckpts/{dataset}/{exp_name}/weights", exist_ok=True)
            os.makedirs(f"ckpts/{dataset}/{exp_name}/images/train", exist_ok=True)
            os.makedirs(f"ckpts/{dataset}/{exp_name}/images/test", exist_ok=True)
            os.makedirs(f"ckpts/{dataset}/{exp_name}/images_best/train", exist_ok=True)
            os.makedirs(f"ckpts/{dataset}/{exp_name}/images_best/test", exist_ok=True)
        else:
            print("Quitting")
            exit(0)

def save_models(netG_toA, netG_toB, dataset, exp_name):
    if isinstance(netG_toA, nn.DataParallel): # we assume all are 
        torch.save(netG_toA.module.state_dict(), f"ckpts/{dataset}/{exp_name}/weights/netG_B2A.pth")
        torch.save(netG_toB.module.state_dict(), f"ckpts/{dataset}/{exp_name}/weights/netG_A2B.pth")
    else:
        torch.save(netG_toA.state_dict(), f"ckpts/{dataset}/{exp_name}/weights/netG_B2A.pth")
        torch.save(netG_toB.state_dict(), f"ckpts/{dataset}/{exp_name}/weights/netG_A2B.pth")
    print("Saved checkpoints")

# Helper to load the right datasets
def load_datasets(datatype, evaluate=False, fraction_time=True):
    ' Loads based on naming convention. should be in a file that matches the datatype given from argparser. dataset class name is always DualDataset '
    dataset_file = importlib.import_module(f"datasets.{datatype}")
    if False:  # evaluate:  
        train_dataset = None   # hack, don't bother loading it if we just need to evaluate
    else:  
        train_dataset = dataset_file.DualDataset("train", evaluate=evaluate, fraction_time=fraction_time)
    test_dataset = dataset_file.DualDataset("test", evaluate=evaluate, fraction_time=fraction_time)

    return train_dataset, test_dataset


# Helper to return the Generator, Discriminator for desired model type
def get_models(modeltype):
    ' Shorthand to import models.cardiogan.Generator, for example. the given modeltype needs to be exactly the file name '
    model_file = importlib.import_module(f'models.{modeltype}')
    if hasattr(model_file, 'Discriminator'):  # hack, if it has Discriminator, then that's the only one and return FreqDiscriminator as none
        return model_file.Generator, model_file.Discriminator, None
    else:
        return model_file.Generator, model_file.TimeDiscriminator, model_file.FrequencyDiscriminator


# Helper to generate all the data from trained models
def generate_signals(batch, device, netG_toA, netG_toB):
    ' Takes in a batch and returns 6 batches: the real data, the generated data, and the (cycle) reconstructed data '
    # This is realA, realB, fakeA, fakeB, recoveredA, recoveredB

    real_image_A = batch["A"].to(device).detach()
    real_image_B = batch["B"].to(device).detach()

    if netG_toA.INPUT_NDIM == 4:
        real_image_A = torch.unsqueeze(real_image_A, 2)
        real_image_B = torch.unsqueeze(real_image_B, 2)

    fake_image_A = netG_toA(real_image_B).detach()
    fake_image_B = netG_toB(real_image_A).detach()

    recovered_image_A = netG_toA(fake_image_B).detach()
    recovered_image_B = netG_toB(fake_image_A).detach()

    # "detach" to drop gradient info, which is only needed for training, and "cpu" to move to cpu
    real_image_A = real_image_A.cpu()
    real_image_B = real_image_B.cpu()
    fake_image_A = fake_image_A.cpu()
    fake_image_B = fake_image_B.cpu()
    recovered_image_A = recovered_image_A.cpu()
    recovered_image_B = recovered_image_B.cpu()

    if netG_toA.INPUT_NDIM == 4:
        " Squish back "
        real_image_A, real_image_B = torch.flatten(real_image_A, 2, 3), torch.flatten(real_image_B, 2, 3)
        fake_image_A, fake_image_B = torch.flatten(fake_image_A, 2, 3), torch.flatten(fake_image_B, 2, 3)
        recovered_image_A, recovered_image_B = torch.flatten(recovered_image_A, 2, 3), torch.flatten(recovered_image_B, 2, 3)

    return real_image_A, real_image_B, fake_image_A, fake_image_B, recovered_image_A, recovered_image_B

def generate_paired_batch(dataset, batch_size, device, netG_toA, netG_toB):
    ' Calls generate_signals but real_A and real_B are from a paired item '

    items = [dataset.get_pair(i) for i in range(batch_size)]
    real_A = torch.stack([torch.Tensor(i['A']) for i in items])
    real_B = torch.stack([torch.Tensor(i['B']) for i in items])

    batch = {"A": real_A, "B": real_B}
    generated_batch = generate_signals(batch, device, netG_toA, netG_toB)
    return generated_batch
    


# Helper to add value to the loss dict
def update_loss_dict(loss_dict, epoch_losses, split):
    ' Moved here just because we do this so many times. Usually done with a Summary object or similar but this works. Add total epoch losses to the loss_dict(list) dict '
    for k, v in epoch_losses.items():
        loss_dict[f"{split}_{k}"].append(v)

    return loss_dict

def update_eval_dict(eval_dict, epoch_eval_metrics, split):
    ' Same as for the loss values '
    for m, v in epoch_eval_metrics.items():
        eval_dict[f"{split}_metric_{m}"].append(v)
    return eval_dict




def get_disc_truth_labels(data_shape, output_shape, device, label_smoothing=False):
    # Returns the 3D or 4D vector that serves as ground truths for discriminators. Returns ones() and zeros() (real and fake)
    # output_shape is from the discriminator -- the shape of the disc output for each signal and lead (so either an int or tuple)

    if label_smoothing:
        alpha, beta = 0.9, 0.1  # NOTE hardcoding this
    else:
        alpha, beta = 1.0, 0.0
        

    batch_size = data_shape[0]
    num_leads = data_shape[1]

    output_shape = (batch_size, num_leads, output_shape) if len(data_shape) == 3 else (batch_size, num_leads, output_shape[0], output_shape[1])

    real_label = Variable(torch.full(output_shape, alpha, dtype=torch.float32), requires_grad=False).to(device)
    fake_label = Variable(torch.full(output_shape, beta, dtype=torch.float32), requires_grad=False).to(device)

    return real_label, fake_label




def pearson(real, recovered):
    ' returns pearson corr for a given sample (real and recovered) '
    real = real.cpu().numpy()
    recovered = recovered.cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ret = 0
        for nl in range(real.shape[0]):  # for each lead separately
            ret += abs(stats.pearsonr(real[nl,:], recovered[nl,:])[0])
    return ret/real.shape[0]  # average across all leads

    
def prd(real, recovered):
    ' percent real difference. Something reported in cardiogan (ablation) '
    return NotImplementedError

def frechet(real, recovered):
    real, recovered = real.cpu().numpy(), recovered.cpu().numpy()
    ret = 0
    if real.shape[0] == 1:
        ret = frdist(real, recovered)
    else:
        for nl in range(real.shape[0]):
            ret += frdist(np.expand_dims(real[nl],0), np.expand_dims(recovered[nl],0))
    return ret/real.shape[0]  # average across leads

def get_average_hr(sig, sig_type, fs):
    ' in bpm, using biosppy '
    try:
        if sig_type == 'ECG' or sig_type == 'EGM' or sig_type == 'SCG':  # SCG just for now
            #ecg = biosppy.signals.ecg.ecg(sig[0], sampling_rate=fs, show=False) # not accurate
            #hrs = ecg['heart_rate']
            try:
                _, res = nk.ecg_peaks(sig[0], sampling_rate=fs, method='hamilton2002')
                peaks = res['ECG_R_Peaks']  # Accurate but slightly slower than py-ecg-detectors.
            except:
                peaks = []
            
        elif sig_type == 'PPG' or sig_type == 'RPPG':
            #ppg = biosppy.signals.ppg.ppg(sig[0], sampling_rate=fs, show=False)
            #hrs = ppg['heart_rate']
            try:
                res = nk.ppg_findpeaks(sig[0].numpy(), sampling_rate=fs)  # sometimes it's so bad it wont work
                peaks = res['PPG_Peaks']
            except:
                peaks = []

        elif sig_type == "ACC":
            peaks = []  # short-circuit, we dont care about heart rate from acc yet

        else:
            raise NotImplementedError

        if len(peaks) == 0:  return 0        
        if len(peaks) == 1:  
            avg_ipi = peaks[0]
        else: 
            avg_ipi = np.mean(np.diff(peaks))
        avg_bpm = (fs*60)/avg_ipi
        return avg_bpm

    except ValueError as e:
        return 0  # Not enough beats detected to compute heart rate -- just say it's 0 

    
def get_hr_mae_for_batch(batchA, batchB, sig_type, fs):
    ' Returns array with diffs of average hr for each sample in batch '
    diffs = []
    for n in range(batchA.shape[0]):
        bpm_A = get_average_hr(batchA[n], sig_type, fs)
        bpm_B = get_average_hr(batchB[n], sig_type, fs)
        mae = abs(bpm_A - bpm_B)
        diffs.append(mae)
    return diffs

# Helper to calculate evaluation metrics
def eval_metrics(data, metric_names=['pearson_real_recovered', 'pearson_real_fake', 'rmse_real_recovered', 'rmse_real_fake',  'frechet', 'hr_mae_real_fake', 'hr_mae_real_recovered'], dataset_metadata=None, last_epoch=False):
    ''' returns a dictionary with {metric_name: sum of vals, one val per sample in batch }. is sum instead of average because this is called on a batch. data is basically a "generate_batch" result; a 6tuple: real_A, real_B, fake_A, fake_B, recovered_A, recovered_B
        dataset_metadata is whatever METADATA is defined by the dataset. Needed for vpeak in particular.
    '''


    if data[0].dim() == 4:
        data = [torch.flatten(d, 1, 2) for d in data]

    # data is a list of the 4 relevant batches:
    # real_image_A, real_image_B, fake_image_A, fake_image_B, recovered_A, recoverd_B = data 
    batch_size = data[0].shape[0]  # real_image_A.shape[0]
    num_leads = (data[0].shape[1], data[1].shape[1])  # real_image_A.shape[1], real_image_B.shape[1]

    metrics = defaultdict(float)  # a dictionary {'pearson': sum([val, val, val, ...]), 'rmse': sum([val, val, val, ...])}

    for i, name in enumerate(["A", "B"]):

       
        real, fake, recovered = data[i].detach(), data[i+2].detach(), data[i+4].detach() # PPG, ECG', PPG'

        for m in metric_names:  # for each metric we want

            if m == 'hr_mae_real_fake':
                ans = sum(get_hr_mae_for_batch(real.cpu(), fake.cpu(), getattr(dataset_metadata, 'NAME_'+name), getattr(dataset_metadata, 'FS_'+name)))  # is an array of each hr diff for each in batch

            elif m == 'hr_mae_real_recovered':
                ans = sum(get_hr_mae_for_batch(real.cpu(), recovered.cpu(), getattr(dataset_metadata, 'NAME_'+name), getattr(dataset_metadata, 'FS_'+name)))  # is an array of the hr diffs

            elif m == 'rmse_real_fake':
                ans = torch.sqrt(torch.square((fake - real)).mean(axis=2))
                ans = ans.sum().item()

            elif m == 'rmse_real_recovered':
                ans = torch.sqrt(torch.square((real - recovered)).mean(axis=2))
                ans = ans.sum().item()

            elif m == 'pearson_real_fake':
                ans = sum([pearson(real[b], fake[b]) for b in range(real.shape[0])])

            elif m == 'pearson_real_recovered':
                ans = sum([pearson(real[b], recovered[b]) for b in range(real.shape[0])])

            elif m == 'prd':
                raise NotImplementedError

            elif m == 'frechet':
                ans = sum([frechet(real[b], recovered[b]) for b in range(real.shape[0])])

            
            metrics[f"{m}_{name}"] = ans  # {m}_{names} will look like 'pearson_A' or 'rmse_B'

    return metrics


