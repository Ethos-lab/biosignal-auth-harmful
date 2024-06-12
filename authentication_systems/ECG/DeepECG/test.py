"""
Basically generate a whole bunch of genuine and imposter pairs, do their matching
thing, and compute EER
"""

import os
import torch
import argparse
from model import DeepECGModel
from dataset import TrainDataset, EvalDataset
from custom_datasets.datasets import get_orig_fs
from torch.utils.data import DataLoader
import numpy as np
from matching import get_binary_template_thresholds, match_real, match_binary
from sklearn.metrics import roc_curve, confusion_matrix
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='should end in /weights/')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--save_stats', action='store_true', help='Saves threshold and max distance to the model pt. Please dont do this with spoofed data')
    parser.add_argument('--match_style', type=str, default='real', choices=['real', 'binary'])
    parser.add_argument('--num_matches', type=int, default=250)
    args = parser.parse_args()

    if args.match_style == 'real':
        matcher = match_real
    elif args.match_style == 'binary':
        matcher = match_binary
   
    #saved_model = torch.load(f"saved_models/{args.train_dataset}.pt") 
    saved_model = torch.load(args.model_path)
    num_classes = saved_model['num_train_classes']  # only used to load the model 

    sig_len = saved_model['sig_len']
    #train_fs = saved_model['fs']
    train_fs = 128  # We hardcoded it, will always be 128
    data_fs = get_orig_fs(args.dataset, spoof=args.spoof_name != 'original', spoof_name=args.spoof_name)
    resample_to = train_fs if train_fs != data_fs else None


    dataset = EvalDataset(args.dataset, spoof=args.spoof_name != 'original', spoof_name=args.spoof_name, split='test')  # split is test  not eval
    # This should be the test split of the train split

    model = DeepECGModel(input_dim=sig_len, output_dim=num_classes)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()

    if args.match_style == 'binary':
        train_dataset = TrainDataset(args.dataset, 'train')  # just to build the match template for match_binary
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        train_threshold = get_binary_template_thresholds(train_loader, model)
    else:
        train_threshold = False

    # See if we can generate real 
    num_genuine, num_imposter = 0, 0
    genuine_generator = dataset.generate_genuine_pair()
    imposter_generator = dataset.generate_imposter_pair()

    # Generate real, imposter datasets and match
    with torch.no_grad():    
        vals, scores = [], []
        distance_reals = []
        for i in range(args.num_matches):
            x1, y1, x2, y2 = next(genuine_generator)
            num_genuine += 1
            vals.append(1)

            distance_real = matcher(x1, x2, model, train_threshold)  # TODO rename the variable
            distance_reals.append(distance_real)

        for i in range(args.num_matches):
            x1, y1, x2, y2 = next(imposter_generator)
            num_imposter += 1
            vals.append(0)
            distance_real = matcher(x1, x2, model, train_threshold)
            distance_reals.append(distance_real)

    print("Num genuine: ", num_genuine, " num_imposter: ", num_imposter)

    # rescale the reals but if binary, then dont need to do this 
    if args.match_style == 'real':
        # Just for now; we have to invert these and scale
        #mnd, mxd = np.min(distance_reals), np.max(distance_reals)
        #scores = [np.abs(1 - (d-mnd)/mxd) for d in distance_reals]
        mxd = np.max(distance_reals)  # Not ideal but what it is
        # Basically want to scale this and then invert it. So largest distance -> 1 and then 1 -> 0, 0 -> 1
        scores = [d/mxd for d in distance_reals]
        scores = [1-d for d in scores]
        #scores = [1/d if d > 0 else 1/mxd for d in distance_reals]  # because roc_curve wants similarity scores i believe
    elif args.match_style == 'binary':
        scores = distance_reals
        scores = -1*np.array(scores)
        mxd = -1

    fpr, tpr, thresholds = roc_curve(vals, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print("EER: ", eer, " at threshold: ", eer_threshold)

    # now acc at that threshold:
    pred_at_thresh = [1 if s > eer_threshold else 0 for s in scores]
    acc = np.mean([1 if a == b else 0 for a, b in zip(vals, pred_at_thresh)])
    print("Acc at eer_thresh: ", acc)
    print("Confusion matrix: ")
    conf_mat = confusion_matrix(vals, pred_at_thresh)
    print(conf_mat)

    if args.save_stats:
        # Add the ere_threshold and distance scale info to the model dict
        saved_model['EER'] = eer  # just so we know
        saved_model['EER_THRESHOLD'] = eer_threshold
        saved_model['MAX_DISTANCE'] = mxd
        if args.spoof_name != 'original':
            print("Youre trying to overwrite stats with spoofed data, what are you doing")
            exit(0)
        torch.save(saved_model, args.model_path)
        print("Resaved the model dict with threshold: ", saved_model['EER_THRESHOLD'], " and distance: ", saved_model['MAX_DISTANCE'])

    # Now on provided threshold
    if 'EER_THRESHOLD' in saved_model:
        threshold = saved_model['EER_THRESHOLD']
        mxd = saved_model['MAX_DISTANCE']

        print()
        print(f"===== Now on found threshold: {threshold} ==== ")

        scores = [d/mxd for d in distance_reals]
        scores = [1-d for d in scores]
        pred_at_thresh = [1 if s > threshold else 0 for s in scores]
        acc = np.mean([1 if a == b else 0 for a, b in zip(vals, pred_at_thresh)])
        print("Acc: ", acc)
        print("Confusion matrix: ")
        conf_mat = confusion_matrix(vals, pred_at_thresh)
        print(conf_mat)
