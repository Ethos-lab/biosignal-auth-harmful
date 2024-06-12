import numpy as np
from torch.utils.data import Dataset
from argparse import ArgumentParser
from custom_datasets import datasets, processing
import os
import random

import signal_processing
from sklearn.metrics import roc_curve

from collections import defaultdict

from scipy.stats import binomtest

"""
python train.py --fs 128 --distance ssim --wavelet morse | tee LOG_OUTPUT_128_SSIM_MORSE.txt



Overall: 
- 1sec segment around peaks **EXTRACTED FROM ECG**
- wavelet transform 
- NN and feature matching
"""

class BaseDataset(Dataset):
    def __init__(self, split, fs=5000, spoof_name='original', wavelet='gabor', time_window=1.0):
        if split == 'train':
            fraction_time = (0.5, 0.625)
        elif split == 'test':
            fraction_time =  (0.625, 0.75)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)  # because it wont fit in memory at 500Hz, change back later
        else:
            raise ValueError

        self.data, self.users = [], []

        # First filtering, then wavelet transform 
        RESAMPLE_FS = fs
        fs_ecg = datasets.get_orig_fs('cebsdb', 'ecg', spoof=False)  # always getting GT ecg
        fs_scg = datasets.get_orig_fs('cebsdb', 'scg', spoof=spoof_name != 'original', spoof_name=spoof_name)
        print("Orig fs for ecg: ", fs_ecg, " and scg: ", fs_scg, " Resampling to: ", RESAMPLE_FS)
        for uix, user, data in datasets.get_orig_spoof_paired_data('cebsdb', 'ecg', 'scg', spoof_name=spoof_name, split='all', fraction_time=fraction_time):
            ecg = data['ecg']
            scg = data['scg']
        
            # Going to resample to 128 to make it easy for us, since they dont care 
            scg = processing.resample_signal(scg, fs_scg, RESAMPLE_FS)[0]
            scg = processing.zscore(scg)
            scg = signal_processing.filter(scg, RESAMPLE_FS)

            _, rpeaks = processing.resample_get_rpeaks(ecg, orig_fs=fs_ecg, resample_to=RESAMPLE_FS)  

            #scg = processing.zscore(scg)  # NOTE remember we did this

            #if spoof_name != 'original':  import pdb; pdb.set_trace()
            ao_peaks = signal_processing.get_AO_peaks(scg, rpeaks, RESAMPLE_FS)
            sigs = signal_processing.segment(scg, ao_peaks, RESAMPLE_FS, time_window=time_window)

            # what if we just do peak detection on this. Much better actually
            #rpeaks = processing.get_ecg_rpeaks_hamilton(scg, RESAMPLE_FS)
            #sigs = signal_processing.segment(scg, rpeaks, RESAMPLE_FS, time_window=time_window)  


            #self.debug1 = scg
            #self.debug2 = ecg
            #self.debug3 = rpeaks
            #self.debug4 = ao_peaks  actually not bad

            #sigs = signal_processing.min_max_norm(sigs)

            # Let's see how this helps, because spoofed is normalized (-1 1)
            #sigs = signal_processing.min_max_norm(sigs)
            #sigs = signal_processing.zscore(sigs)
            sigs = signal_processing.detrend_each(sigs)
            sigs = signal_processing.wavelet_transform(sigs, type=wavelet, fs=RESAMPLE_FS)

            
            # Note: for dep elearning, "we compress the wavelet transformed SCG image to a size of 80x80"
    
            self.data.extend(sigs)


            self.users.extend([uix for x in range(len(sigs))])

            #print(f"Processed data for user: {user}, shape: {len(sigs)}")
            print(".", end='', flush=True)

        #self.data = np.concatenate(self.data, 0)  # when single dim
        self.data = np.stack(self.data, 0)  # now 2D
        print('', flush=True)
    

    def get_data_for_user(self, user):
        subset = [d for i, d in enumerate(self.data) if self.users[i] == user]
        return np.stack(subset)

    def get_data_for_other_users(self, user):
        subset = [d for i, d in enumerate(self.data) if self.users[i] != user]
        return np.stack(subset)

    def get_rand_sample(self, user):
        # randomly get a sample that's either from this user or not from this user
        same = bool(random.getrandbits(1))
        if same:
            subset = [d for i, d in enumerate(self.data) if self.users[i] == user]
        else:
            subset = [d for i, d in enumerate(self.data) if self.users[i] != user]
        data2 = random.choice(subset)
        return data2, same

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.users[ix]



def get_eer(scores, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr-fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr-fpr))]

    # acc at this threshold? 
    y_pred = []
    for s in scores:
        if s > eer_threshold:  y_pred.append(1)
        else:  y_pred.append(0)
    acc = np.mean(np.equal(y_true, y_pred))

    return eer, eer_threshold, acc

def predict(template, distance_score, eer_thresh, subset, example_ixs):
    y_pred = []
    for ex in example_ixs:
        test_sample = subset[ex]
        
        score = distance_score(template, test_sample)
        if score >= eer_thresh:  y_pred.append(True)
        else:  y_pred.append(False)
    return y_pred, np.mean(y_pred)


def eval(args, distance_score, EER_THRESH, test_dset, eval_dset, baseline=False):
    # Get pairs between test and eval 
    ACC_OVER_TRIALS = defaultdict(list)
    ALL_Y_PRED = defaultdict(list)
    users = list(range(len(EER_THRESH)))
    for u in users:
        # First get the original user template
        subset = test_dset.get_data_for_user(u)
        template = np.mean(subset, 0)  # template to compare against 
        eer_thresh = EER_THRESH[u]

        # now subset eval --> these are all the potential spoofed samples we could send
        if baseline:
            user_subset = eval_dset.get_data_for_user(u)
            num_needed = len(user_subset)
            subset = test_dset.get_data_for_other_users(u)
            random.shuffle(subset)
            subset = subset[:num_needed]  # sample the same amount as we would have before
            print("Available: ", num_needed)
        else:
            user_subset = eval_dset.get_data_for_user(u)
        subset_ixs = set(list(range(len(subset))))
        example_ixs = random.sample(list(subset_ixs), k=min(args.num_matches, len(subset_ixs)))
        remaining = set(subset_ixs).difference(example_ixs) 
       
        for i in range(args.num_trials):
            y_pred, acc = predict(template, distance_score, eer_thresh, subset, example_ixs)
            ACC_OVER_TRIALS[i].append(acc)
            ALL_Y_PRED[i].extend(y_pred) 
            # filter examples
            for yi, y in enumerate(y_pred):
                if y:  continue 
                else:
                    if len(remaining) == 0:  continue  # no more to choose, plateau here
                    new_ix = random.choice(list(remaining))
                    example_ixs[yi] = new_ix
                    remaining = remaining - set([new_ix])
            
    for i in range(args.num_trials):
        #print(np.mean(ACC_OVER_TRIALS[i]), "\t", np.std(ACC_OVER_TRIALS[i]))
        y_pred = ALL_Y_PRED[i]
        bt = binomtest(np.sum(y_pred), args.num_matches*len(users))
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")



def train(args):
    """
    Conclusions so far: ssim is the best distance metric
    So far morse (scales in (0.01, 1.0, 64) better than gabor with 2sec segemtns). Honestly just go with it)
    """

    # "for the distacne matching approach, we get the average of the training data for a given user and use that as the template. 
    train_dset = BaseDataset(split='train', fs=args.fs, wavelet=args.wavelet, time_window=args.time_window)
    test_dset = BaseDataset(split='test', fs=args.fs, wavelet=args.wavelet, time_window=args.time_window) # original 

    users = set(train_dset.users)

    if args.distance == 'ssim':
        distance_score = signal_processing.ssim
    elif args.distance == 'manhattan' or args.distance == 'l1':
        distance_score = lambda feat1, feat2:  -np.linalg.norm(feat2-feat1, ord=1)
    elif args.distance == 'l2':
        distance_score = lambda feat1, feat2: - np.linalg.norm(feat2-feat1, ord=2)  # these should maybe be RMSE and not RMS, to standardize
    else:
        raise NotImplementedError

    y_true, scores = [], []
    all_eers = []
    all_accs = []
    EER_THRESH = []  # need to store this to use in eval later
    for u in users:
        subset = train_dset.get_data_for_user(u)
        
        template = np.mean(subset, 0)
        # And save this 

        # Eval against randomly sampled from test_dset
        y_true, scores = [], []
        for i in range(500):
            test_sample, same = test_dset.get_rand_sample(u)
            y_true.append(same)
            score = distance_score(template, test_sample)
            scores.append(score)

        eer, eer_thresh, acc = get_eer(scores, y_true)
        all_eers.append(eer)
        all_accs.append(acc)
        EER_THRESH.append(eer_thresh)
        print(f"For user: {u}, Test set EER: {eer:.2f}, EER_THRESH: {eer_thresh:.4f}, ACC: {100*acc:.2f}%")

    print(f"Average of EERs: {np.mean(all_eers):.4f} and average of ACCs: {np.mean(all_accs):.2f}")

    print("="*25)
    # Now eval: 
    print("First eval original")
    eval(args, distance_score, EER_THRESH, test_dset, test_dset)
    print("Now eval " + args.spoof_name)

    eval_dset = BaseDataset(split='eval', fs=args.fs, wavelet=args.wavelet, time_window=args.time_window, spoof_name=args.spoof_name)
    eval(args, distance_score, EER_THRESH, test_dset, eval_dset)

    eval_dset = BaseDataset(split='eval', fs=args.fs, wavelet=args.wavelet, time_window=args.time_window, spoof_name='original')
    eval(args, distance_score, EER_THRESH, test_dset, eval_dset, baseline=args.baseline)


def debug(args):
    eval_dset = BaseDataset(split='eval', fs=args.fs, wavelet=args.wavelet, time_window=args.time_window, spoof_name=args.spoof_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--fs', type=int, default=128, help='fs to resample to')
    parser.add_argument('--wavelet', type=str, default='morse', help='morse, gabor, bump')
    parser.add_argument('--distance', type=str, default='l2', help='ssim, manhattan, l2')
    parser.add_argument('--time_window', type=float, default=1.0, help='Seconds for each segment, centered around peak')
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--num_matches', type=int, default=25)  # note this is pe r patient
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    train(args)

    # Hmm waht if we train at 5000Hz but eval at 128 Hz? 
