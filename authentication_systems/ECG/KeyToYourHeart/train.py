import argparse
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, f1_score
import time
import random
import pickle as pkl

def split_data(dataset, patient_ix, r=1, train_r=0.6):
    # So dataset is an [] with all patients data. patient_ix ones should be pos; otherwise neg 
    # rebalance pct means that the number of 0s/imposters i rebalance*num_genuines. So 1 is 100% is the same number

    train_x, train_y = [], []
    test_x, test_y = [], []

    for dix, d in enumerate(dataset):
        num_train = round(train_r*len(d))  # train and CV on the first 80% of the signals
        num_test = len(d) - num_train

        train_x.extend(d[:num_train,:])
        test_x.extend(d[num_train:,:])  

        if dix != patient_ix:
            train_y.extend([0 for i in range(num_train)])
            test_y.extend([0 for i in range(num_test)])
        else:
            train_y.extend([1 for i in range(num_train)])
            test_y.extend([1 for i in range(num_test)])


    train_x, train_y, test_x, test_y = rebalance(train_x, train_y, test_x, test_y, r)

    return train_x, train_y, test_x, test_y        

def rebalance(train_x, train_y, test_x, test_y, r=1.0):
    # Duplicate samples so there's an equal (or r) amound of genuine v imposter samples, by random sampling

    # first train then test
    genuines = [i for i, x in enumerate(train_y) if x == 1]
    num_imposters = r * len(genuines)
    subset_imposters = [i for i, x in enumerate(train_y) if x == 0]
    choose_subset_imposters = random.sample(subset_imposters, num_imposters)
    train_y = [x for i, x in enumerate(train_y) if i in choose_subset_imposters or i in genuines]
    train_x = [x for i, x in enumerate(train_x) if i in choose_subset_imposters or i in genuines]

    genuines = [i for i, x in enumerate(test_y) if x == 1]
    num_imposters = r * len(genuines)
    subset_imposters = [i for i, x in enumerate(test_y) if x == 0]
    choose_subset_imposters = random.sample(subset_imposters, num_imposters)
    test_y = [x for i, x in enumerate(test_y) if i in choose_subset_imposters or i in genuines]
    test_x = [x for i, x in enumerate(test_x) if i in choose_subset_imposters or i in genuines]
   
    return train_x, train_y, test_x, test_y

def train_model(train_x, train_y, kernel='rbf', c=1.0):
    " Trains an SVM " 
    c = 1.0
    model = SVC(kernel=kernel, C=c, gamma='scale', class_weight='balanced', probability=True)
    model.fit(train_x, train_y)
    return model

def test_model(model, test_x, test_y):
    " Tests, evals EER "
    y_pred = model.predict(test_x)
    y_scores = model.predict_proba(test_x)[:,1]

    pct_ones = sum(y_pred)/len(y_pred)

    f1 = f1_score(test_y, y_pred)

    fpr, tpr, threshold = roc_curve(test_y, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return f1, EER, eer_threshold, pct_ones

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train, after calling generate_data')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_name', type=str)

    # These were set; just added for hcitagging debugging 
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='rbf')
    args = parser.parse_args()


    # First read and split data  in 60/40 as they do 
    read_path = 'path/to/processed/data'
    patients = os.listdir(read_path)


    # Load all patients data first
    data = []
    for pix, patient in enumerate(patients):
        fn = os.path.join(read_path, patient)
        patient_data = np.load(fn)
        data.append(patient_data)

    # Now train each model 
    EERS = []
    for pix, patient in enumerate(patients):

        train_x, train_y, test_x, test_y = split_data(data, pix)  # 80/20 for hyperparameter tuning, 100/0 for the final model to save


        # Train a model: 
        model = train_model(train_x, train_y, args.kernel, args.c)
        train_f1, train_eer, train_eer_thresh, train_pct_ones = test_model(model, train_x, train_y)
        test_f1, test_eer, test_eer_thresh, test_pct_ones = test_model(model, test_x, test_y)
        print(f"Patient {patient} \t" + 
            f"Num Train: {len(train_y)} " +
            f"Num Test: {len(test_y)} " +
            f"Train F1: {train_f1:2f} " +
            f"TrainEER: {train_eer:.2f} " +
            f"TrainPctOnes: {train_pct_ones:.2f} " +
            f"Test F1: {test_f1:.2f} " +
            f"EER: {test_eer:.2f} " + 
            f"PctOnes: {test_pct_ones:.2f} " + 
            f"EERThresh: {test_eer_thresh:.3f} ")
        EERS.append(test_eer)

        if args.save:
            if not args.save_name:
                save_name = args.dataset
            else:
                save_name = args.dataset + "_" + args.save_name

            model_save_dir = os.path.join('saved_models', save_name,  patient.split('.')[0])
            os.makedirs(model_save_dir, exist_ok=True)
            state_dict = {
                'model': model,
                'EER': test_eer,
                'EER_THRESHOLD': test_eer_thresh
            }
            with open(os.path.join(model_save_dir, 'model.pkl'), 'wb') as fp:
                pkl.dump(state_dict, fp)
                print("Saved to: ", fp.name)

    print()
    print(f"Average Test EER: {np.mean(EERS)*100:.2f}")
