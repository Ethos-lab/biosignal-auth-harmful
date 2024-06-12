"""
Helper functions for saving images
The 'main' function that gets imported in other files is "plot_and_save". This switches on the dataset type and calls a bunch of other plotting functions defined here. Can add/delete those calls as needed
"""

import matplotlib.pyplot as plt
import os
import wandb


def plot_and_save(batch, file_name, send_wandb=False):
    '''
    Main function that gets imported/called in other files
    Plots relevent figures after training for given dataset and saves. 
    Assumes batch is a list with the 6 batches (output of 'generate_signals' function in training file

    Changed a bunch -- now plots a single (paired) set of data on top of eachother. 
    
    Plot real and generated data on top of eachother
    If we have paired data, this is two sets of overlaps: realA/recoveredA and realB/fakeB
    Without paired data, this is realA/recoveredA and fakeB

    Note: Pared back version just for RPPG that doesn't have ground truth:

    real_A, real_B, fake_A, fake_B, recovered_A, recovered_B = batch

    plot_paired_batch([real_A, fake_B, real_B, recovered_A], os.path.join(file_name, 'batch_A'), send_wandb=send_wandb)
    plot_paired_batch([real_B, fake_A, real_A, recovered_B], os.path.join(file_name, 'batch_B'), send_wandb=send_wandb)

    '''

    real_image_A, real_image_B, fake_image_A, fake_image_B, recovered_image_A, recovered_image_B = batch

    # Plot the orig, fake, and recovered images independently: 
        
    plot_and_save_signal(real_image_A, file_name+"/real_A", "Orig source", send_wandb)
    plot_and_save_signal(real_image_B, file_name+"/real_B", "Orig target", send_wandb)
    plot_and_save_signal(fake_image_A, file_name+"/fake_A", "Fake source", send_wandb)
    plot_and_save_signal(fake_image_B, file_name+"/fake_B", "Fake target", send_wandb)
    plot_and_save_signal(recovered_image_A, file_name+"/recovered_A", "Recovered source", send_wandb)
    plot_and_save_signal(recovered_image_B, file_name+"/recovered_B", "Recovered target", send_wandb)

    # Also plot the orig and recovered signals on top of each other:
    plot_orig_recovered(real_image_A, recovered_image_A, file_name+"/orig_recovered_A", "", send_wandb)
    plot_orig_recovered(real_image_B, recovered_image_B, file_name+"/orig_recovered_B", "", send_wandb)

    # Also plot the orig and fake signals in the same figure:
    plot_source_target(real_image_A, fake_image_B, file_name+"/source_target_A", "", send_wandb)
    plot_source_target(real_image_B, fake_image_A, file_name+"/source_target_B", "", send_wandb)



def plot_paired_batch(batch, filename_prefix, send_wandb=False):

    real_A, fake_B, real_B, recovered_A = batch

    num_in_batch = real_A.shape[0]
    num_leads_A = real_A.shape[1]
    num_leads_B = real_B.shape[1]

    num_leads = num_leads_A + num_leads_B

    for i in range(num_in_batch):
        fig, ax = plt.subplots(num_leads, 1)

        # plot real_A and recovered_A on top of eachother
        for a in range(num_leads_A):
            ax[a].plot(real_A[i,a,:])
            ax[a].plot(recovered_A[i,a,:])
            ax[a].set_xticks([])
        a = num_leads_A
        for b in range(num_leads_B):
            if real_B is not None:  ax[a+b].plot(real_B[i,b,:])
            ax[a+b].plot(fake_B[i,b,:])
            ax[a+b].set_xticks([])

        name = f"{filename_prefix}_{i}.png"
        plt.savefig(name)
        if send_wandb:  wandb.log({name: fig})
        plt.close()
        print(f'Saved image {name} of {num_in_batch}')
    
   
 


def plot_and_save_signal(batch, filename_prefix, title=None, send_wandb=False):
    ' Plots all leads of input signal and saves to filename_prefix+batch_num for each in batch. Optional title '
    num_in_batch = batch.shape[0]
    num_leads = batch.shape[1]
    for i in range(num_in_batch):
        fig, ax = plt.subplots(num_leads, 1)
        if num_leads == 1:  ax = [ax]
        if title is not None:  plt.title(title)
        for a in range(num_leads):
            ax[a].plot(batch[i,a,:])
            if a < num_leads-1:  ax[a].set_xticks([])
        name = f"{filename_prefix}_{i}.png"
        plt.savefig(name)
        if send_wandb:  wandb.log({name: fig})
        plt.close()
        print(f"Saved image {name} of {num_in_batch}")


def plot_orig_recovered(batch_orig, batch_recovered, filename_prefix, title=None, send_wandb=False):
    ' Plots orig and recovered signals on top of each other; different png for each in the batch. Optional title'
    num_in_batch = batch_orig.shape[0]
    num_leads = batch_recovered.shape[1]
    for i in range(num_in_batch):
        fig, ax = plt.subplots(num_leads, 1)
        if num_leads == 1:  ax = [ax]
        if title is not None:  plt.title(title)
        for a in range(num_leads):
            ax[a].plot(batch_orig[i,a,:])
            ax[a].plot(batch_recovered[i,a,:])
            if a < num_leads-1:  ax[a].set_xticks([])
        plt.legend(['Original', 'Recovered'])
        name = f"{filename_prefix}_{i}.png"
        plt.savefig(name)
        if send_wandb:  wandb.log({name: fig})
        plt.close()
        print(f'Saved image {name} of {num_in_batch}')

def plot_source_target(batch_source, batch_target, filename_prefix, title=None, send_wandb=False):
    ' Plots source and target data above/below each other in the same figure. Optional title '
    num_in_batch = batch_source.shape[0]
    num_leads_source = batch_source.shape[1]
    num_leads_target = batch_target.shape[1]
    num_leads = num_leads_source + num_leads_target
    for i in range(num_in_batch):
        fig, ax = plt.subplots(num_leads, 1)
        if num_leads == 1:  ax = [ax]
        if title is not None:  plt.title(title)
        for a in range(num_leads_source):
            ax[a].plot(batch_source[i,a,:])
            ax[a].set_xticks([])
        for a in range(num_leads_source, num_leads):
            ax[a].plot(batch_target[i,a-num_leads_source,:], c='r')
            if a < num_leads-1:  ax[a].set_xticks([])

        name = f"{filename_prefix}_{i}.png"
        plt.savefig(name)
        if send_wandb:  wandb.log({name: fig})
        plt.close()
        print(f'Saved image {name} of {num_in_batch}')
    

