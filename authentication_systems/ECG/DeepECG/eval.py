""" Custom eval, prob success over tries at the orig test eer_thresh """

import argparse 
import numpy as np
import os
import torch
from dataset import DualEvalDataset, TrainDataset
from matching import match_real, match_binary, get_binary_template_thresholds
from model import DeepECGModel
from collections import defaultdict

from torch.utils.data import DataLoader

from scipy.stats import binomtest

def mydictgettr(keys, d):
    # This was getting annoying
    res = []
    for k in keys:
        attr = getattr(d, k)
        res.append(attr)
    return res


def create_matches(pair_generator, num_matches=100):
    # Assume that for each in sigs1, sigs2, the corresponding each in users is the same
    examples = []
    pair_generator = dataset.generate_genuine_pair()
    for i in range(num_matches):
        res = next(pair_generator)
        examples.append(res)

    return examples


def filter_matches(dataset, examples, bool_correct):
    """
    If a pair succeeded in passing authentication, keep it
    Otherwise, replace with another random pair of (orig data, spoofed data)
    """
    already_tried = [(e['ix1'], e['ix2']) for e in examples]
   
    for ie, e in enumerate(examples):
        if not bool_correct[ie]: 
            # Replace this ix in exmaples with an ew pair from datset.generate_pair()
            ix1 = e['ix1']
            ix2 = e['ix2']
            
            max_time_searching = 0
            while ix2 == ix2 and (ix1, ix2) in already_tried:  
                max_time_searching += 1
                res = dataset.get_another_pair(e['ix1'])  # given e[ix1] generate another. Note this will go through all we got already
                ix1, ix2 = res['ix1'], res['ix2']

                if max_time_searching > 1000:  
                    #res = None
                    break

            examples[ie] = res

    # clear up all the Nones: 
    #examples = [e for e in examples if e is not None] # let's see what chaos this causes

    return examples
        
        
def predict(model, threshold, max_distance, examples, matcher, input_noise=0, train_medians=None):

    distances = []
    for i, e in enumerate(examples):
        #noise = np.random.random(e['sig1'].size)*input_noise
        #noise = np.expand_dims(noise, 0)
        #noise = np.array([n - (input_noise/2) for n in noise])
        sig1 = e['sig1']
        sig2 = e['sig2']

        dist = matcher(sig1, sig2, model, train_medians)
        
        distances.append(dist)

    if train_medians is not None:  # being lazy and using this to mean match_style is binary
        scores = -1*np.array(distances)
    else:
        scores = [d/max_distance for d in distances]
        scores = [1-d for d in scores]

    y_pred = [1 if s > threshold else 0 for s in scores]
    acc = np.mean(y_pred)

    return y_pred, acc
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='pt file, needs to have eer_thresh saved')
    parser.add_argument('--dataset', type=str, default='original')
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--num_matches', type=int, default=500, help='for both pos and neg')
    parser.add_argument('--num_repeats', type=int, default=50, help='Repeat initial selection 50 times')
    parser.add_argument('--match_style', type=str, default='real', choices=['real', 'binary'])
    parser.add_argument('--input_noise', type=float, default=0)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_per_pat', action='store_true')
    args = parser.parse_args()

    print("Match style chosen: ", args.match_style)


    saved_model = torch.load(args.model_path)
    num_classes = saved_model['num_train_classes']
    sig_len = saved_model['sig_len'] # unused now
    train_fs = 128 if not 'fs' in saved_model else saved_model['fs']  # also kinda unused cause it'll always be 128

    dataset = DualEvalDataset(args.dataset, spoof_name1='original', spoof_name2=args.spoof_name, baseline=args.baseline)

    model = DeepECGModel(input_dim=sig_len, output_dim=num_classes)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()

    if args.match_style == 'real':
        matcher = match_real
    elif args.match_style == 'binary':
        matcher = match_binary
    else:
        raise ValueError

    if 'EER' in saved_model:  
        eer = saved_model['EER']
        print("Loaded model with EER: ", eer)  # just fyi
    threshold = saved_model['EER_THRESHOLD']
    max_distance = saved_model['MAX_DISTANCE']

    if args.match_style == 'binary':
        # prob should have saved the train distance median array in saved_model too, dammit. anyway, recreate it here
        train_dataset = TrainDataset(args.dataset, 'train')  # just to build the match template for match_binary
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        train_threshold = get_binary_template_thresholds(train_loader, model)
    else:
        train_threshold = None

    
    ACC_OVER_ATTEMPTS = []  # to hold data for attempts that we then average and report results
    STD_OVER_ATTEMPTS = []
    num_repeats = args.num_repeats  # number of times we do this so that our results have a +/- std
    num_attempts = 10 # will get 10 numbers at the end of this that hopefully go up to 100%
    num_matches = args.num_matches

    examples = create_matches(dataset, num_matches=num_matches)
    for i in range(num_attempts):
        
        y_pred, acc = predict(model, threshold, max_distance, examples, matcher, args.input_noise, train_threshold)

        bt = binomtest(np.sum(y_pred), num_matches)
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")
        examples = filter_matches(dataset, examples, y_pred)
