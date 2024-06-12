import argparse
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, f1_score
import time
import random
import pickle as pkl
from collections import defaultdict
from scipy.stats import binomtest
"""
Basically the same as test but we chose a random one of the provided dataset and see if it works
If not, chose another random one
Keep going up to 10
Average over patients
"""

def test_model(model, test_x, test_y, eer_threshold):
    " Tests, evals EER "
    y_scores = model.predict_proba(test_x)[:,1]
    y_pred = [1 if y > eer_threshold else 0 for y in y_scores]

    pct_ones = sum(y_pred)/len(y_pred)

    return y_pred, pct_ones

def filter_examples(y_pred, random_selection_order, list_ix_subset, sample_size):
    # Returns a list of <sample_size> len of indices into the large array y_pred, where each time we yield a new selection.
    # This is annoyingly convoluted, prob a better way to code this, idk
    # filter_examples(y_pred, random_selection_order, random_selection, sample_size)

    # list_ix_subset is the current list of 10 indices into y_pred
    new_list_ixs = []

    if len(list_ix_subset) == 0:
        if sample_size > len(random_selection_order):  
            sample_size = len(random_selection_order)
            print("Adjusting sample size to: ", len(random_selection_order))
        for li in range(sample_size):
            ne = random_selection_order.pop(0)
            new_list_ixs.append(ne)
    else:
        for li in list_ix_subset:
            if y_pred[li]:  new_list_ixs.append(li)  # it passed, keep it 
            else:
                # pop the next off random_seletion_order 
                if len(random_selection_order) == 0:  # no more remaining, stay with what we have 
                    new_list_ixs.append(li)
                else:
                    ne = random_selection_order.pop(0)
                    new_list_ixs.append(ne)

    return new_list_ixs

def run_attempt(all_data, patient, model, eer_thresh, baseline, baseline_pat):
            
    # Remember patient is pix here

    num_this_user = all_data[patient].shape[0]

    Y_PRED_FOR_PAT = defaultdict(list)

    if baseline:

        if baseline_pat:
            data = [a for ai, a in enumerate(all_data) if ai == baseline_pat]
        else:
            data = [a for ai, a in enumerate(all_data) if ai != patient]
    else:
        data = [a for ai, a in enumerate(all_data) if ai == patient]

    # Truncate to least we have 
    data = np.concatenate(data, 0)
    random.shuffle(data)
    data = data[:num_this_user,:]

    targets = [1 for i in range(len(data))]
    y_pred, pct_ones = test_model(model, data, targets, eer_thresh)  # just get all results now
    len_total = len(y_pred)

    random_selection_order = list(range(len_total))
    random.shuffle(random_selection_order)
    random_selection = []  # 'examples'
    for i in range(num_trials):
        random_selection = filter_examples(y_pred, random_selection_order, random_selection, sample_size)
        yp = [y_pred[r] for r in random_selection]
        Y_PRED_FOR_PAT[i].extend(yp)

    return Y_PRED_FOR_PAT


if __name__ == "__main__":
    parser = argparse.ArgumentParser('test, get eer thresh and ppv using that')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_dir', type=str, default='/saved_models/dataset/')
    parser.add_argument('--spoof_name', type=str, default='original')  # Default firs spoof_name is original. So default is eval on self
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_per_pat', action='store_true')
    args = parser.parse_args()


    # First read and spit data 
    read_path = os.path.join('/media', 'big_hdd', 'kveena', 'key-to-heart', 'monday', args.dataset, 'test', args.spoof_name)
    patients = os.listdir(read_path)

    sample_size = 50  # send in 10 at a time, basically the same as num_redos, averaging of n times
    num_trials = 10 
    #num_repeats = 10
    total_num_samps = 0  # just for printing/reporting
    #AVERAGE_OVER_PATIENTS = [0]*num_trials
    #STD_OVER_PATIENTS = [0]*num_trials
    Y_PRED_ALL_PATIENTS = defaultdict(list)  # hold the actual y_pred vals
    print(args.dataset, args.spoof_name)

    all_data = []
    #if args.baseline:
    #    # load other data
    for pix, patient in enumerate(patients):
        fn = os.path.join(read_path, patient)
        data = np.load(fn)
        all_data.append(data)

    for pix, patient in enumerate(patients):

        # For each patient, send the spoofed data in over n trials and see when it gets accepted

        # Load model 
        with open(os.path.join(args.model_dir, patient.split('.')[0], "model.pkl"), "rb") as fp:
            model = pkl.load(fp)
        eer_thresh = model['EER_THRESHOLD']
        model = model['model']


        if args.baseline:
            if args.baseline_per_pat:
                # Loop over all the others -
                for bix, base_pat in enumerate(patients):
                    if bix == pix: continue
                    y_pred_for_pix_and_bix = run_attempt(all_data, pix, model, eer_thresh, args.baseline, bix)
                    for i in y_pred_for_pix_and_bix.keys():
                        Y_PRED_ALL_PATIENTS[i].extend(y_pred_for_pix_and_bix[i])
            else:
                y_pred_for_pix_and_bix = run_attempt(all_data, pix, model, eer_thresh, args.baseline, None)
                for i in y_pred_for_pix_and_bix:  Y_PRED_ALL_PATIENTS[i].extend(y_pred_for_pix_and_bix[i])
        else:
            y_pred_for_pix_and_bix = run_attempt(all_data, pix, model, eer_thresh, False, None)
            for i in y_pred_for_pix_and_bix:  Y_PRED_ALL_PATIENTS[i].extend(y_pred_for_pix_and_bix[i])

    for i in range(num_trials):
        yp = Y_PRED_ALL_PATIENTS[i]
        bt = binomtest(np.sum(yp), len(yp))
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(yp):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")

