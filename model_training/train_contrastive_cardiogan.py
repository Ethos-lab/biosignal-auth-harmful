import numpy as np
import torch
import torch.nn as nn
import os
import models
import datasets
from torch.utils.data import DataLoader
import argparse
import itertools
import random 
import torch.backends.cudnn as cudnn
from models.model_utils import DecayLR, ReplayBuffer, weights_init
import time
from matplotlib import pyplot as plt
import pickle
from math import ceil
from collections import defaultdict
import torch.nn.functional as F
import wandb
import utils
from plot_helpers import plot_and_save
import json
import copy

from contrastive_losses import SupConLoss, SimCLRLoss
from models.yhcardiogan import IdentityEncoder

parser = argparse.ArgumentParser(
    description='Unpaired bio-signal translation with contrastive loss')

# Logging, printing, saving options
parser.add_argument("--exp_name", default="debug", type=str, help="Will save log and ckpts in a directory with this name, as ckpts/<datasetname>/<exp_name>")
parser.add_argument('--force', '-f', action='store_true', help='Rewwrites existing log/ckpt directory without asking to confirm')
parser.add_argument("-p", "--print_freq", default=50, type=int, metavar="N", help="print frequency. (default:50)")
parser.add_argument('--num_to_plot', type=int, default=4, help='number of plots of each kind to save after done training')
parser.add_argument('--wandb', action='store_true', help='Send metrics to wandb')
parser.add_argument('--metrics_every_epoch', action='store_true', help='If true, calculates all test metrics at every epochs, instead of just at the end. Otherwise just calculates RMSE every epoch, and does plots/metrics at the last epoch. Takes more time')
# Dataset options
parser.add_argument('--dataset', type=str, required=True, help='Name of dataset file. ie.. ecgppg_bidmc')
parser.add_argument("-b", "--batch_size", default=128, type=int, help='batch size, default 128')

# Model options
parser.add_argument('--model_type', default='yhcardiogan', choices=['cyclegan', 'cardiogan', 'yhcardiogan'])

# Training options
parser.add_argument("--cpu", action="store_true", help="Runs on cpu")
parser.add_argument("--epochs", default=25, type=int, help="number of total epochs to run. cardiogan does 15 but seeing improvements up to 25")
parser.add_argument("--decay_epochs", type=float, default=0.67, help="fraction of total epochs to start decaying linearly to 0 (default was 50/200, which is why this is 0.25. Cardiogan does 10/15, so should be 0.67 I guess)")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate. (default:0.0001). Cardiogan uses 1e-4")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--both_gpus', action='store_true', help='uses both gpus with dataparallel')

# Hyperparams for weighting the different losses
parser.add_argument('--alpha', default=3, type=int, help='coeff for time disc loss, default 3')
parser.add_argument('--beta', default=1, type=int, help='coeff for freq disc loss, 1')
parser.add_argument('--gamma', default=30, type=int, help='coeff for cycle loss. cardiogan paper uses lambda and 30')
parser.add_argument('--zeta', default=0.1, type=float, help='coeff for contrastive loss')


# Other experimental options (not used in final version)
parser.add_argument('--eta', default=0, type=float, help='coeff for hr loss')
parser.add_argument('--gan_loss', type=str, choices=['mse', 'bce'], default='bce')
parser.add_argument('--cycle_loss', type=str, choices=['l1', 'mse'], default='l1')
parser.add_argument('--con_loss', type=str, choices=['supcon', 'simclr'], default='supcon')
parser.add_argument('--no_replay_buffer', action='store_true', help='Default swtich losses')
parser.add_argument('--final_sigmoid', action='store_true', help='For Discs, add a final sigmoid. NOTE cant do this for bce loss because it already does it')
parser.add_argument('--disc_out_1d', action='store_true', help='For Discs, if output squished to 1 dimensional')
parser.add_argument('--representation', type=str, choices=['generator_output', 'generator_bottleneck'], default='generator_bottleneck')
parser.add_argument('--plot_each_time', action='store_true')
parser.add_argument('--sweep', action='store_true')
parser.add_argument('--save_best', type=str, choices=['loss_total_gan', 'rmse_real_fake_A'])

""" Example usage:

python train_cardiogan.py --dataset ecgppg_cardiogan --epochs 15 


"""






if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    # Make exp_name checkpoint directory -- exist_okay only if confirmed
    utils.make_dirs(args.dataset, args.exp_name, args.force)  

    # Monitor training progress with wandb
    if args.wandb:  
        wandb.init(project="<project-name>", entity="<org-name>")
        wandb.config.update(args)
        wandb.run.name = args.dataset + "-" + args.exp_name

    
    # Wandb offers 'sweep' for hyperparameter search
    if args.sweep:
        if not args.wandb:  raise NotImplementedError('sweep only with wandb')
        sweep_configuration = {
            'method': 'grid',
            'metric': {'name': 'real_fake_rmse', 'goal': 'minimize'},
            'parameters': {
                    'alpha': {'values': []},
                    'beta': {'values': []},
                    'gamma': {'values': []},
                    'zeta': {'values': []}
                }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='cyclegan-biometrics')

    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True  # for faster performance

    device = torch.device("cuda:"+str(args.gpu)) if torch.cuda.is_available() and not args.cpu else torch.device("cpu")  # this file needs to be shorter
    if not args.both_gpus:  
        print('Using device: ', device, torch.cuda.get_device_name(device))

    # Loss weight terms for GAN, cyclic losses
    alpha, beta, gamma = args.alpha, args.beta, args.gamma



    ####################################
    #   Datasets, loaders
    train_dataset, test_dataset = utils.load_datasets(args.dataset)
    num_workers = 8 if args.both_gpus else 4
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    print("Dataset metdata:")
    print(train_dataset.METADATA)
    ####################################



    ####################################
    #  Models    
    DO_FREQ_DISC = args.model_type.find('cardiogan') > -1  # Flag for doing dual (time and freq based) discriminators
    if DO_FREQ_DISC:  print("Using dual-discriminator approach")
    Generator, Discriminator, FrequencyDiscriminator = utils.get_models(args.model_type)
    netG_toA = Generator(train_dataset.METADATA.NUM_LEADS_B, train_dataset.METADATA.NUM_LEADS_A).to(device).apply(weights_init)
    netG_toB = Generator(train_dataset.METADATA.NUM_LEADS_A, train_dataset.METADATA.NUM_LEADS_B).to(device).apply(weights_init)
    netD_A = Discriminator(train_dataset.METADATA.NUM_LEADS_A, final_sigmoid=args.final_sigmoid, out_1d=args.disc_out_1d).to(device).apply(weights_init)
    netD_B = Discriminator(train_dataset.METADATA.NUM_LEADS_B, final_sigmoid=args.final_sigmoid, out_1d=args.disc_out_1d).to(device).apply(weights_init)
    if DO_FREQ_DISC:  
        netD_A_FREQ = FrequencyDiscriminator(train_dataset.METADATA.NUM_LEADS_A, final_sigmoid=args.final_sigmoid, out_1d=args.disc_out_1d).to(device).apply(weights_init)
        netD_B_FREQ = FrequencyDiscriminator(train_dataset.METADATA.NUM_LEADS_B, final_sigmoid=args.final_sigmoid, out_1d=args.disc_out_1d).to(device).apply(weights_init)
    print("Defined models and initialized weights")

    do_4dim = Generator.INPUT_NDIM == 4
    print("Using 4D data approach")

    if args.both_gpus:
        print("Using 2 GPUs")
        netG_toA, netG_toB = nn.DataParallel(netG_toA, device_ids=[0,1]), nn.DataParallel(netG_toB, device_ids=[0,1])
        netD_A, netD_B = nn.DataParallel(netD_A, device_ids=[0,1]), nn.DataParallel(netD_B, device_ids=[0,1])
        if DO_FREQ_DISC:
            netD_A_FREQ, netD_B_FREQ = nn.DataParallel(netD_A_FREQ, device_ids=[0,1]), nn.DataParallel(netD_B_FREQ, device_ids=[0,1])


    # Default usses a replay buffer to make GAN training more stable. 
    NO_REPLAY_BUFFER = args.no_replay_buffer

    # Contrastive loss option (unused in final version)
    netIE = IdentityEncoder(1).to(device)
    ####################################


    ####################################
    #  Define loss function (adversarial_loss) and optimizer based on parser args
    #if args.switch_losses:
    if args.gan_loss == 'mse':
        adversarial_loss = torch.nn.MSELoss().to(device)
    elif args.gan_loss == 'bce':
        adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        raise NotImplementedError("gan loss type: " + str(args.gan_loss))

    if args.cycle_loss == 'l1':
        cycle_loss = torch.nn.L1Loss().to(device)
    elif args.cycle_loss == 'mse':
        cycle_loss = torch.nn.MSELoss().to(device)
    else:
        raise NotImplementedError("cycle loss type: " + str(args.cycle_loss))

    if args.con_loss == 'supcon':
        con_loss = SupConLoss()
    elif args.con_loss == 'simclr':
        con_loss = SimCLRLoss(device=device)
    else:
        raise NotImplementedError('contrastive loss type: ' + str(args.con_loss))

    # Now the (optional) hr loss  (default weight is 0; unused in final version)
    hr_loss = torch.nn.MSELoss().to(device)

    print(f"Defined adversarial_loss: {adversarial_loss} and cycle_loss {cycle_loss}")
    ####################################
    

    ####################################
    #  Optimizers
    lr = args.lr  # default is 1e-4, what cardiogan paper says
    optimizer_G = torch.optim.Adam(itertools.chain(netG_toB.parameters(), netG_toA.parameters()),
                                   lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))
    if DO_FREQ_DISC:
        optimizer_D_A_FREQ = torch.optim.Adam(netD_A_FREQ.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D_B_FREQ = torch.optim.Adam(netD_B_FREQ.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(itertools.chain(netG_toB.parameters(), netG_toA.parameters()),
                                    lr=lr, betas=(0.5, 0.999))  # separate opt for contrastive loss
    optimizer_HR = torch.optim.Adam(netG_toA.parameters(), lr=lr, betas=(0.5, 0.999))
    print("Created optimizers")
    ####################################


    ####################################
    # LR Schedulers
    if args.epochs - ceil(args.decay_epochs*args.epochs) <= 0:  args.decay_epochs = 0  # when epochs is small (under 2), which we do for debugging, make this not error
    lr_lambda = DecayLR(args.epochs, 0, ceil(args.decay_epochs*args.epochs)).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda, verbose=True)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda, verbose=True)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda, verbose=True)
    if DO_FREQ_DISC:
        lr_scheduler_D_A_FREQ = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A_FREQ, lr_lambda=lr_lambda, verbose=True)
        lr_scheduler_D_B_FREQ = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B_FREQ, lr_lambda=lr_lambda, verbose=True)
    print("Created learning rate schedulers")
    ####################################


    # Dicts hold all training and testing lossses, metrics (keys appended by 'training.loss.etc' for example). Gets written as json at end of each epoch
    loss_dict = defaultdict(list)
    eval_dict = defaultdict(list)

    # init best_metric_so_far to know when to save the model 
    best_metric_so_far = np.inf
    best_epoch = 0

    #start train function here
    for epoch in range(0, args.epochs):


        print(f".... Starting epoch {epoch} ....")

        # Starting train:
        netG_toA.train()
        netG_toB.train()
        netD_A.train()
        netD_B.train()
        if DO_FREQ_DISC:
            netD_A_FREQ.train()
            netD_B_FREQ.train()
        
        epoch_losses = defaultdict(float)  # losses for this epoch -- eventually added to loss_dict with prefix "train"
        epoch_metrics = defaultdict(float)  # eval metrics for this epoch -- eventually added to eval_dict with prefix "train"

        fake_A_buffer = ReplayBuffer()  
        fake_B_buffer = ReplayBuffer()

        et0 = time.time()
        
        for i, data in enumerate(train_dataloader):
            it0 = time.time()

            real_A = data['A'].to(device)  # shape: batch_size x num_leads x sample_len
            real_B = data['B'].to(device)
            person_label_A = data['LABEL_A'].float().to(device)  # for con loss on A->B netG_toB
            person_label_B = data['LABEL_B'].float().to(device)  # etc 

            if do_4dim:
                real_A = real_A.unsqueeze_(2)  # shape: batch_size x num_leads x 1 x sample_len
                real_B = real_B.unsqueeze_(2)
            
            # Abstracted away the creation of 6 arrays that serve as the comparisons for each disc
            # real_label and fake_label is ones() or zeros() of size (batch_size x num_leads x 1 x 16) if INPUT_DIM == 4 or (batch_size x num_leads) if INPUT_DIM==3
            # real_label_freq and fake_label_freq has the last dims of (4, 4)
            real_label_A, fake_label_A = utils.get_disc_truth_labels(real_A.shape, netD_A.OUTPUT_SHAPE, device)
            real_label_B, fake_label_B = utils.get_disc_truth_labels(real_B.shape, netD_B.OUTPUT_SHAPE, device)
            if DO_FREQ_DISC:
                real_label_freq_A, fake_label_freq_A = utils.get_disc_truth_labels(real_A.shape, netD_A_FREQ.OUTPUT_SHAPE, device)
                real_label_freq_B, fake_label_freq_B = utils.get_disc_truth_labels(real_B.shape, netD_B_FREQ.OUTPUT_SHAPE, device)
                
            
            # =====================================================
            # Stage 1: generate fake signals, update generators
            # =====================================================

            optimizer_G.zero_grad() # same type of optimizer for the netG_toA and netG_toB

            # Forward Pass
            fake_A = netG_toA(real_B)
            fake_B = netG_toB(real_A)
            recovered_A = netG_toA(fake_B)
            recovered_B = netG_toB(fake_A)

            # Loss #1/13: Generator loss, A, time
            fake_output_A = netD_A(fake_A)  # the predicted disc output from the fake_A
            loss_GAN_A_time = adversarial_loss(fake_output_A, real_label_A)  # generated A should look real 
            epoch_losses['loss_1_GAN_A_time'] += loss_GAN_A_time.item()

            # 2/13: Generator loss, B, time
            fake_output_B = netD_B(fake_B)
            loss_GAN_B_time = adversarial_loss(fake_output_B, real_label_B)
            epoch_losses['loss_2_GAN_B_time'] += loss_GAN_B_time.item()
            

            # 3/13: Generator loss, A, freq
            # 4/13: Generator loss, B, freq 
            if DO_FREQ_DISC:
                fake_output_A = netD_A_FREQ(fake_A)
                fake_output_B = netD_B_FREQ(fake_B)

                loss_GAN_A_freq = adversarial_loss(fake_output_A, real_label_freq_A)
                epoch_losses['loss_3_GAN_A_freq'] += loss_GAN_A_freq.item()
           
                loss_GAN_B_freq = adversarial_loss(fake_output_B, real_label_freq_B)
                epoch_losses['loss_4_GAN_B_freq'] += loss_GAN_B_freq.item()

            # 5/13: Cycle-consistent loss, A
            recovered_A = netG_toA(fake_B)
            loss_cycle_ABA = cycle_loss(recovered_A, real_A)
            epoch_losses['loss_5_cycle_ABA'] += loss_cycle_ABA.item()

            # 6/13: Cycle-consistent loss, B
            recovered_B = netG_toB(fake_A)
            loss_cycle_BAB = cycle_loss(recovered_B, real_B)
            epoch_losses['loss_6_cycle_BAB'] += loss_cycle_BAB.item()

            # Combined GAN losses
            gan_loss = alpha*(loss_GAN_A_time + loss_GAN_B_time) + gamma*(loss_cycle_ABA + loss_cycle_BAB)
            if DO_FREQ_DISC:
                gan_loss += beta*(loss_GAN_A_freq + loss_GAN_B_freq)
            epoch_losses['loss_total_GAN'] += gan_loss.item()

            # Update the generators together
            gan_loss.backward()
            optimizer_G.step()


            # =====================================================
            # Stage 2: Update the discriminators
            # =====================================================
    
            if not NO_REPLAY_BUFFER:        
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                fake_B = fake_B_buffer.push_and_pop(fake_B)


            # 7/13: time a discr train
            optimizer_D_A.zero_grad() 
            loss_disc_time_real_A = adversarial_loss(netD_A(real_A), real_label_A)
            loss_disc_time_fake_A = adversarial_loss(netD_A(fake_A.detach()), fake_label_A)
            loss_disc_time_A = 0.5*(loss_disc_time_real_A + loss_disc_time_fake_A)
            loss_disc_time_A.backward()
            epoch_losses['loss_7_D_A_time'] += loss_disc_time_A.item()
            optimizer_D_A.step()

            # 8/13: time b discr train
            optimizer_D_B.zero_grad() 
            loss_disc_time_real_B = adversarial_loss(netD_B(real_B), real_label_B)
            loss_disc_time_fake_B = adversarial_loss(netD_B(fake_B.detach()), fake_label_B)
            loss_disc_time_B = 0.5*(loss_disc_time_real_B + loss_disc_time_fake_B)
            loss_disc_time_B.backward()
            epoch_losses['loss_8_D_B_time'] += loss_disc_time_B.item()
            optimizer_D_B.step()

            # 9/13: freq a discr train
            # 10/13: freq b discr train
            if DO_FREQ_DISC:
                optimizer_D_A_FREQ.zero_grad()
                loss_disc_freq_real_A = adversarial_loss(netD_A_FREQ(real_A), real_label_freq_A)
                loss_disc_freq_fake_A = adversarial_loss(netD_A_FREQ(fake_A.detach()), fake_label_freq_A)
                loss_disc_freq_A = 0.5*(loss_disc_freq_real_A + loss_disc_freq_fake_A)
                loss_disc_freq_A.backward()
                epoch_losses['loss_9_D_A_freq'] += loss_disc_freq_A.item()
                optimizer_D_A_FREQ.step()

                optimizer_D_B_FREQ.zero_grad()
                loss_disc_freq_real_B = adversarial_loss(netD_B_FREQ(real_B), real_label_freq_B)
                loss_disc_freq_fake_B = adversarial_loss(netD_B_FREQ(fake_B.detach()), fake_label_freq_B)
                loss_disc_freq_B = 0.5*(loss_disc_freq_real_B + loss_disc_freq_fake_B)
                loss_disc_freq_B.backward()
                epoch_losses['loss_10_D_B_freq'] += loss_disc_freq_B.item()
                optimizer_D_B_FREQ.step()

            """
            # 11/13: Contrastive loss on the bottleneck of the generator
            """
            # Update generator: either using an additional encoding network after generated signal, or on bottleneck and just 8D
            if args.zeta > 0:
                optimizer_C.zero_grad()
                label_mask = torch.eq(person_label_A.view(-1,1), person_label_A.view(-1,1).T)
                label_mask = label_mask.fill_diagonal_(0).sum(1)

                if args.representation == 'generator_output':
                    raise NotImplementedError # I dont think this was working, get rid of it for now
                elif args.representation == 'generator_bottleneck':
                    identity_encoding = netG_toB.get_bottleneck(real_A)  # Identity here means 'person', not like f(x)=x
                else:
                    raise NotImplementedError
                loss_con_A = con_loss(identity_encoding, label_mask)


                """
                # 12/13: Con loss on the other direction (B->A), netG_toA
                """
                label_mask = torch.eq(person_label_B.view(-1,1), person_label_B.view(-1,1).T)
                label_mask = label_mask.fill_diagonal_(0).sum(1)

                identity_encoding = netG_toA.get_bottleneck(real_B)
                loss_con_B = con_loss(identity_encoding, label_mask)

            
                # Combined A and B --
                loss_con = loss_con_A + loss_con_B
                loss_con *= args.zeta
                epoch_losses['loss_11_contrastive'] += loss_con.item()
                loss_con.backward()
                optimizer_C.step()
            else:
                loss_con = 0


            # Loss 13/13 - on HR. This will be very slow. Was just for experimenting; unused in final version
            if args.eta > 0: # dont bother if we dont need it 
                optimizer_HR.zero_grad()
                loss_HR = hr_loss(real_A, fake_A)
                loss_HR *= args.zeta
                epoch_losses['loss_mse_hr'] += loss_HR.item()
                loss_HR.backward()
                optimizer_HR.step()
                


           ##############################################
            # Monitoring, saving, printing progress, etc
            #############################################
                
            disc_loss = epoch_losses['loss_7_D_A_time'] + epoch_losses['loss_8_D_B_time']
            if DO_FREQ_DISC:  disc_loss += epoch_losses['loss_9_D_A_freq'] + epoch_losses['loss_10_D_B_freq'] 
            epoch_losses['loss_total_disc'] += disc_loss
           
            if args.metrics_every_epoch or epoch == args.epochs-1:
                iter_metrics = utils.eval_metrics([real_A, real_B, fake_A, fake_B, recovered_A, recovered_B], dataset_metadata=train_dataset.METADATA)
            else:  # still do rmse because it's quick
                iter_metrics = utils.eval_metrics([real_A, real_B, fake_A, fake_B, recovered_A, recovered_B], metric_names=['rmse_real_fake', 'rmse_real_recovered'], dataset_metadata=train_dataset.METADATA) 
            for k, v in iter_metrics.items():  epoch_metrics[k] += v  # keep a running sum of each metric across minibatches

            
            print(
                 f"[{epoch}/{args.epochs - 1}][{i}/{len(train_dataloader) - 1}] "
                 f"loss_gan: {gan_loss:.3f} "
                 f"loss_disc: {disc_loss:.3f} "
                 f"loss_con: {loss_con:.3f} "
                 f"\t Took: {time.time()-it0:.3f}s")

        
        # End of training epoch metrics
            
        epoch_metrics = {k: v/len(train_dataloader.dataset) for k, v in epoch_metrics.items()}
        loss_dict = utils.update_loss_dict(loss_dict, epoch_losses, 'train')
        eval_dict = utils.update_eval_dict(eval_dict, epoch_metrics, 'train')
        

        # Print metrics on whole data set for this epoch:
        print("Train losses for epoch: ", end='')
        print(
            f"loss_total_GAN: {epoch_losses['loss_total_GAN']:.3f} "
            f"loss_total_disc: {epoch_losses['loss_total_disc']:.3f}")

        print("Train metrics: ", end='')
        for k, v in epoch_metrics.items():  print(f"{k}: {v:.4f}", end=' ')
        print()
        print(f"Training Epoch {epoch} took: {time.time()-et0:.3f}s")
            
            









        ##################################################################
        # Now evaluate: 
        #
        #
        #

        with torch.no_grad():

            netG_toA.eval()
            netG_toB.eval()
            netD_A.eval()
            netD_B.eval()
            if DO_FREQ_DISC:
                netD_A_FREQ.eval()
                netD_B_FREQ.eval()

            epoch_losses = defaultdict(float) # Running sum of losses per epoch -- at end of epoch gets added to 'loss_dict' with prefix 'test'
            epoch_metrics = defaultdict(float) # Running sum of metrics per epoch -- at end, gets added to 'eval_dict' with prefix 'test'

            fake_A_buffer = ReplayBuffer()
            fake_B_buffer = ReplayBuffer()

            tt0 = time.time() # start of testing this epoch
            for i, batch in enumerate(test_dataloader):

                real_A = batch['A'].to(device)
                real_B = batch['B'].to(device)

                if do_4dim:
                    real_A, real_B = real_A.unsqueeze_(2), real_B.unsqueeze_(2)  # data is batch x leads x 1 x 512 now

                real_label_A, fake_label_A = utils.get_disc_truth_labels(real_A.shape, netD_A.OUTPUT_SHAPE, device)
                real_label_B, fake_label_B = utils.get_disc_truth_labels(real_B.shape, netD_B.OUTPUT_SHAPE, device)
                if DO_FREQ_DISC: 
                    real_label_freq_A, fake_label_freq_A = utils.get_disc_truth_labels(real_A.shape, netD_A_FREQ.OUTPUT_SHAPE, device)
                    real_label_freq_B, fake_label_freq_B = utils.get_disc_truth_labels(real_B.shape, netD_B_FREQ.OUTPUT_SHAPE, device)


                fake_A = netG_toA(real_B)
                fake_output_A = netD_A(fake_A)

                loss_GAN_A_time = adversarial_loss(fake_output_A, real_label_A)
                epoch_losses['loss_1_GAN_A_time'] += loss_GAN_A_time.item()

                fake_B = netG_toB(real_A)
                fake_output_B = netD_B(fake_B)
                loss_GAN_B_time = adversarial_loss(fake_output_B, real_label_B)
                epoch_losses['loss_2_GAN_B_time'] += loss_GAN_B_time.item()


                if DO_FREQ_DISC:
                    fake_output_A = netD_A_FREQ(fake_A)
                    loss_GAN_A_freq = adversarial_loss(fake_output_A, real_label_freq_A)
                    epoch_losses['loss_3_GAN_A_freq'] += loss_GAN_A_freq.item()

                    fake_output_B = netD_B_FREQ(fake_B)
                    loss_GAN_B_freq = adversarial_loss(fake_output_B, real_label_freq_B)
                    epoch_losses['loss_4_GAN_B_freq'] += loss_GAN_B_freq.item()

                recovered_A = netG_toA(fake_B)
                loss_cycle_ABA = cycle_loss(recovered_A, real_A)
                epoch_losses['loss_5_cycle_ABA'] += loss_cycle_ABA.item()

                recovered_B = netG_toB(fake_A)
                loss_cycle_BAB = cycle_loss(recovered_B, real_B)
                epoch_losses['loss_6_cycle_BAB'] += loss_cycle_BAB.item()

                gan_loss = alpha*(loss_GAN_A_time + loss_GAN_B_time) + gamma*(loss_cycle_ABA + loss_cycle_BAB)
                if DO_FREQ_DISC:  gan_loss += beta*(loss_GAN_A_freq + loss_GAN_B_freq)

                epoch_losses['loss_total_GAN'] += gan_loss.item()

                # Now the discriminators
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                fake_B = fake_B_buffer.push_and_pop(fake_B)

                loss_disc_time_real_A = adversarial_loss(netD_A(real_A), real_label_A)
                loss_disc_time_fake_A = adversarial_loss(netD_A(fake_A.detach()), fake_label_A)
                loss_disc_time_A = 0.5*(loss_disc_time_real_A + loss_disc_time_fake_A)
                epoch_losses['loss_7_D_A_time'] += loss_disc_time_A.item()

                loss_disc_time_real_B = adversarial_loss(netD_B(real_B), real_label_B)
                loss_disc_time_fake_B = adversarial_loss(netD_B(fake_B.detach()), fake_label_B)
                loss_disc_time_B = 0.5*(loss_disc_time_real_B + loss_disc_time_fake_B)
                epoch_losses['loss_8_D_B_time'] += loss_disc_time_B.item()

                
                if DO_FREQ_DISC:  
                    loss_disc_freq_real_A = adversarial_loss(netD_A_FREQ(real_A), real_label_freq_A)
                    loss_disc_freq_fake_A = adversarial_loss(netD_A_FREQ(fake_A.detach()), fake_label_freq_A)
                    loss_disc_freq_A = 0.5*(loss_disc_freq_real_A + loss_disc_freq_fake_A)
                    epoch_losses['loss_9_D_A_freq'] += loss_disc_freq_A.item()

                    loss_disc_freq_real_B = adversarial_loss(netD_B_FREQ(real_B), real_label_freq_B)
                    loss_disc_freq_fake_B = adversarial_loss(netD_B_FREQ(fake_B.detach()), fake_label_freq_B)
                    loss_disc_freq_B = 0.5*(loss_disc_freq_real_B + loss_disc_freq_fake_B)
                    epoch_losses['loss_10_D_B_freq'] += loss_disc_freq_B.item()


                epoch_losses['loss_11_contrastive'] += 0 
               
                disc_loss = epoch_losses['loss_7_D_A_time'] + epoch_losses['loss_8_D_B_time']
                if DO_FREQ_DISC:  disc_loss += epoch_losses['loss_9_D_A_freq'] + epoch_losses['loss_10_D_B_freq'] 
                epoch_losses['loss_total_disc'] += disc_loss



                # In addition to loss values, also calculate a variety of eval metrics
                if args.metrics_every_epoch or epoch == args.epochs-1:
                    iter_metrics = utils.eval_metrics([real_A, real_B, fake_A, fake_B, recovered_A, recovered_B], dataset_metadata=train_dataset.METADATA)
                else:  # only do rmse
                    iter_metrics = utils.eval_metrics([real_A, real_B, fake_A, fake_B, recovered_A, recovered_B], metric_names=['rmse_real_fake', 'rmse_real_recovered'], dataset_metadata=train_dataset.METADATA) 
                for k, v in iter_metrics.items():  epoch_metrics[k] += v

            # Print total epoch loss metrics for the test set:
            print(
                f"Test losses: "
                f"loss_total_gan: {epoch_losses['loss_total_GAN']:.4f} "
                f"loss_total_disc: {epoch_losses['loss_total_disc']:.4f} "
                )



        # Log and upload metrics

        epoch_metrics = {k: v/(len(test_dataloader.dataset)) for k, v in epoch_metrics.items()}

        if args.wandb:
            wandb.log({"test.losses": epoch_losses})
            wandb.log({"test.metrics": epoch_metrics})

        loss_dict = utils.update_loss_dict(loss_dict, epoch_losses, 'test')
        eval_dict = utils.update_eval_dict(eval_dict, epoch_metrics, 'test')

        print("Test metrics for epoch: ", end='')
        for k, v in epoch_metrics.items():  print(f"{k}: {v/len(test_dataset):.4f}", end=' ')
       

        #############
        # At the end of an epoch: 
        #  - write out loss_dict and eval_dict
        #  - update lr schedulers
        #  - save generators (dont need discs yet)
        #  - optionally plot images
        #  - optionally keep track of the best model so far, save it somewhere 
 
        # also save the loss dict and the eval metric dict
        with open(os.path.join("ckpts", args.dataset, args.exp_name, "loss_dict.json"), "w") as fp:
            json.dump(loss_dict, fp)
        with open(os.path.join("ckpts", args.dataset, args.exp_name, "eval_dict.json"), "w") as fp:
            json.dump(eval_dict, fp)

        if args.save_best:
            if args.save_best == 'rmse_real_fake_A':
                this_metric = np.mean(epoch_metrics['rmse_real_fake_A'])
            elif args.save_best == 'loss_total_gan':
                this_metric = epoch_losses['loss_total_gan']
            else:  raise NotImplementedError # i mean it shouldnt be able to get here anyway

            # Only save this if this is best so far 
            if this_metric < best_metric_so_far: 
                print("Improvement -- keeping this model")
                best_metric_so_far = this_metric
                best_models = (copy.deepcopy(netG_toA), copy.deepcopy(netG_toB))
                best_epoch = epoch
        else:
            # Default, just save the most recent 
            best_epoch = epoch
            best_models = (netG_toA, netG_toB)

        # save generators, dont bother saving discriminators? 
        utils.save_models(best_models[0], best_models[1], args.dataset, args.exp_name)
            

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        if DO_FREQ_DISC:
            lr_scheduler_D_A_FREQ.step()
            lr_scheduler_D_B_FREQ.step()

        
        # After finished all epochs, plot and save the first 8 images from the test, train datasets (unless you want it every epoch)
        if ((epoch == args.epochs - 1) and (args.num_to_plot > 0)) or (args.plot_each_time):
            train_batch = utils.generate_paired_batch(train_dataset, args.num_to_plot, device, netG_toA, netG_toB)
            plot_and_save(train_batch, os.path.join("ckpts", args.dataset, args.exp_name, "images", "train"), args.wandb)

            test_batch = utils.generate_paired_batch(test_dataset, args.num_to_plot, device, netG_toA, netG_toB)
            plot_and_save(test_batch, os.path.join("ckpts", args.dataset, args.exp_name, "images", "test"), args.wandb)

        print()
        print("="*25)

    print("Finished training. Best epoch ", best_epoch)
    print("Updating plots for models from best epoch...")
    train_batch = utils.generate_paired_batch(train_dataset, args.num_to_plot, device, best_models[0], best_models[1])
    plot_and_save(train_batch, os.path.join("ckpts", args.dataset, args.exp_name, "images_best", "train"), args.wandb)

    test_batch = utils.generate_paired_batch(test_dataset, args.num_to_plot, device, best_models[0], best_models[1])
    plot_and_save(test_batch, os.path.join("ckpts", args.dataset, args.exp_name, "images_best", "test"), args.wandb)

