from datasets import TrainDataset

from argparse import ArgumentParser
import os
import numpy as np

from custom_datasets import datasets, processing

import random

from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

import pickle

class EvalDataset:
    def __init__(self, dset1, dset2, both_bcg_ecg=False, num_comps=500):
        """ Basically get random pairs of each, then there's a matching """
        self.dset1 = dset1
        self.dset2 = dset2
        self.num_comps=num_comps
        
        # for convenience:
        self.num_ixs = len(self.dset1)
        self.unique_users = list(set(self.dset1.PATIENT_IX))

    def __len__(self):
        return self.num_comps

    def __getitem__(self, ix):
        # Get a pair of data from dset1, dset2, and whether they're a match. Try to balance so use prob = 50

        rand_ix1 = random.randint(0, self.num_ixs-1)
        data1, user1 = self.dset1[rand_ix1]

        same = bool(random.getrandbits(1))
        if same:
            subset = [i for i, u in enumerate(self.dset2.PATIENT_IX) if u == user1 and i < len(self.dset2)]
        else:
            subset = [i for i, u in enumerate(self.dset2.PATIENT_IX) if u != user1 and i < len(self.dset2)]
        rand_ix2 = random.choice(subset)
        if rand_ix2 >= len(self.dset2):  import pdb; pdb.set_trace()
        data2, _ = self.dset2[rand_ix2]
        return data1, data2, float(same)

def eval_eer(model, test_dset, y_true_u):
    scores_u = model.predict_proba(test_dset.data)
    #scores_u = scores[:,u]
    #y_true_u = [1 if t == u else 0 for t in test_dset.users]
    #import pdb; pdb.set_trace()
    fpr, tpr, thresholds = roc_curve(y_true_u, scores_u[:,1], pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr-fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr-fpr))]

    y_pred = np.argmax(scores_u, axis=1)  # want 1 to be associated with the max one
    y_pred_thresh = [1 if y>= eer_threshold else 0 for y in scores_u[:,1]]
    acc = np.equal(y_true_u, y_pred_thresh).mean()
     
    return eer, eer_threshold, acc

def train(args):

    train_dset = TrainDataset(split='train')
    test_dset = TrainDataset(split='test')
    
    # Now evaluate: 
    ALL_EER = []
    ALL_EER_THRESH = []
    for u in set(train_dset.users):
        model = SVC(probability=True, C=args.C)
        y_true = [1 if y == u else 0 for y in train_dset.users]
        model.fit(train_dset.data, y_true)

        #model = SVC(probability=True, C=args.C)  # I assume they're doing a multiclass SVM?
        #model.fit(train_dset.data, train_dset.users)  # This will probably take forever

        y_true = [1 if y == u else 0 for y in test_dset.users]
        eer, eer_thresh, acc = eval_eer(model, test_dset, y_true)
        ALL_EER.append(eer)
    
        # Debug: metrics on train train set, cause this makes no sense - 
        print(f"User {u}\tEER: {eer}, EER_THRESH: {eer_thresh}, ACC: {acc:.2f}")


        if args.save:
            print("="*50)
            savedir = os.path.join('saved_models', args.save)
            if not os.path.exists(savedir):  os.makedirs(savedir)
            savefn = os.path.join(savedir, f'{u}.pkl')
            with open(savefn, 'wb') as fp:
                save_dict = {
                    'model': model,
                    'EER_THRESH': eer_thresh,
                    'EER': eer
                }
                pickle.dump(save_dict, fp)
            
                print('Saved to: ', savefn)

    print("="*25)
    print("Average EER: ", np.mean(ALL_EER))


    # and now eval on test and eval dataset  using matching/l2 distance, pairs etc 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--save', type=str, default='', help='Save filename, default not to save')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--both', action='store_true', help='Add in ecg')
    args = parser.parse_args()
    train(args)
