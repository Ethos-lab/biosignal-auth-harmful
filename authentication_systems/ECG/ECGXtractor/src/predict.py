import argparse
import pandas as pd
import os
import json
import numpy as np
import util
import loader
import tqdm
from tensorflow.keras import models, metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, mean_absolute_error
import sys
import util

# Added by Veena: These needed to load the saved keras networks, which didnt work ootb
import glob
import tensorflow as tf
from keras import backend as K

# Added by Veena: Report EER for train and validation 
from custom_eer_metric import EER

MAX_BS = 64


def predict(params, seed_, model_path=None):
    preproc = None
    if model_path is not None:
        preproc = util.load(os.path.dirname(model_path))
    print("Loading testing set...")
    test = loader.prepare_dataset(params['test'], params)
    print("Test size: " + str(len(test['ecg'])) + " examples.")


    if params['experiment'].find("verification") >= 0:
        #positive_samples = params.get('positive_samples', 250)
        #negative_multiplier = params.get('negative_multiplier', 1)
        positive_samples = 1000  # 50/50 for getting EER   # TODO maybe should be what we trained on
        negative_multiplier = 1
        num_samples = positive_samples + (positive_samples * negative_multiplier)
        batch_size = params.get("batch_size", 16) if num_samples % params.get("batch_size", 16) == 0 else \
            util.get_best_bs(num_samples, MAX_BS)
        test_examples = util.create_generic_match_sample(preproc, test, positive_samples, negative_multiplier, seed_)
        test_len = len(test_examples)
        test_gen = loader.match_sample_generator(batch_size, test_examples)
        print("Should be testing num samples: ", num_samples)

    else:
        input_keys_json = params['input_keys_json']
        output_keys_json = params['output_keys_json']
        if params['experiment'].find('identification') >= 0:
            test = util.fix_data(test)
        test_len = len(test['ecg'])
        batch_size = params.get("batch_size", 16) if test_len % params.get("batch_size", 16) == 0 else \
            util.get_best_bs(test_len, MAX_BS)
        test_gen = loader.data_generator(batch_size, preproc, test, input_keys_json, output_keys_json)

    if params['experiment'].find("verification") >= 0:
        y_true = np.array([])
        for tg in test_gen:  # THIS IS AN IDIOTIC WAY TO DO THIS. IF THIS DOESNT DEPLETE THE GENERATOR, THIS WONT ACTUALLY MATCH THE Y_PRED
            y_true = np.hstack((y_true, tg[1][:, 1] if tg[1].shape[1] == 2 else tg[1][0]))
            if len(y_true) >= num_samples:
                break
    else:
        y_true = np.array([])
        for tg in test_gen:
            if len(y_true) == 0:
                y_true = y_true.reshape(-1, tg[1].shape[1])
            y_true = np.vstack((y_true, tg[1]))
            if len(y_true) >= test_len:
                break


    inputs = np.stack([b['match_sample'] for b in test_examples])
    y_true = np.array([b['is_matching'] for b in test_examples])

    # Veena: annoying - some models load the first way, some only the second
    model = models.load_model(model_path, custom_objects={"K": K, "EER": EER})
    print("Loaded model: ", model_path)

    import math
    #y_pred = model.predict(test_gen, verbose=1, steps=math.ceil(test_len / batch_size))
    y_pred = model.predict(inputs, verbose=1)

    if params['experiment'].find('verification') >= 0:
        y_pred_acc = np.argmax(y_pred, axis=1)
        # y_pred_acc = np.array([1 if e[1] > 0.1338 else 0 for e in y_pred])
        y_pred = y_pred[:, 1]  #scores 

        # y_true = y_true[:len(y_pred)]
        print("AT THRESH 0.5: ")
        print("AUC: ", roc_auc_score(y_true, y_pred))
        print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred_acc))
        print("Accuracy: ", accuracy_score(y_true, y_pred_acc))
        print("Negative samples: ", len(np.where(y_true == 0)[0]))
        print("Positive samples: ", len(np.where(y_true == 1)[0]))
        print("anyway....\n")


        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        fnr = 1 - tpr
        eer_thresh = threshold[np.nanargmin(np.absolute(fnr - fpr))]
        eer_ = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        print("EER: ", eer_, " Threshold: ", eer_thresh)
        print("At this eer thresh, confusion mat:")
        y_pred_acc_at_eer_thresh = np.array([1 if e > eer_thresh else 0 for e in y_pred])
        print(confusion_matrix(y_true, y_pred_acc))

        #eer_, eer_thresh = eer_functions(fpr, tpr, threshold, y_true, y_pred)  # redoing this - dont need a separate call
        
        return eer_, eer_thresh
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        np.set_printoptions(threshold=sys.maxsize)
        print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))
        print("Accuracy: ", accuracy_score(y_true, y_pred))

    print("Len test examples: ", test_len)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name')
    parser.add_argument('--spoof_name', help='dataset spoof name', default='original')
    parser.add_argument('--save_stats', action='store_true', help='Saves EER Thresh to same dir as model_path as an npy')
    # Following all have defaults
    parser.add_argument("--config_file", help="path to config file, has a default", default='configs/siamese/dataset/config_test_spoof_name')
    parser.add_argument("--model_name", help="Load model at saved_<dataset>/verification_<dataset>/<model_name>/<model_name>.hdf5. Default is 'latest'", default='latest')
    parser.add_argument('--num_trials', type=int, default=1)
    args = parser.parse_args()

    dataset = args.dataset
    assert dataset
    assert args.num_trials == 1, "We changed this, do you know hwat youre doing?"
    model_path = os.path.join(f'saved_{dataset}', f'verification_{dataset}', args.model_name, f'{args.model_name}.hdf5')

    params = json.load(open(args.config_file, 'r'))
    if params['experiment'].find('verification') >= 0:
        eer_list, eer_thresh_list = [], []
        for i in range(args.num_trials):  # TODO why are we not just doing it on the entire dataset? 
            eer, eer_thresh = predict(params, i, model_path)  # what why are we combingint hese twho things?
            eer_list.append(eer)
            eer_thresh_list.append(eer_thresh)
        print("EERS: ", eer_list)
        print("EER THRESHS: ", eer_thresh_list)
        print()
        avg_eer = np.mean(eer_list)
        avg_eer_thresh = np.mean(eer_thresh_list)
        print(f"Average EER: {avg_eer * 100 :.2f}")
        print(f"Average Thresh: {avg_eer_thresh : .6f}")
    else:
        print("Not verification? ")
        #predict(params, 2, args.model_path)  # Veena uncommenting this
        raise ValueError

    if args.save_stats:
        save_dir = os.path.dirname(model_path)
        statsfile = os.path.join(save_dir, 'EER_THRESH.npy')
        statsfileeer = os.path.join(save_dir, 'EER.npy')  # adding this in 
        if os.path.exists(statsfile):
            print("File ", statsfile, " exists. Want to rewrite? (Yes/Y/y)")
            out = input().lower()
            if out != 'yes' and out != 'y':
                exit(0)
    
        np.save(statsfile, avg_eer_thresh)
        np.save(statsfileeer, avg_eer)
        print("Wrote to file: ", statsfile)
        print("Wrote to file: ", statsfileeer)


    """
    def eer_functions(fpr, tpr, threshold, y_true, y_pred):
    import matplotlib.pyplot as plt
    eer_th = threshold[np.argmin(np.absolute(fpr - (1 - tpr)))]
    eer = (fpr[np.argmin(np.absolute(fpr - (1 - tpr)))] + (1 - tpr[np.argmin(np.absolute(fpr - (1 - tpr)))])) / 2
    # plt.plot(threshold, fpr)
    # plt.plot(threshold, 1 - tpr)
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))
    # plt.show()
    print("EER: ", eer, " - Threshold: ", eer_th)
    y_pred_eer = np.array([1 if e > eer_th else 0 for e in y_pred])
    print("Confusion Matrix at EER Thresh: \n", confusion_matrix(y_true, y_pred_eer))
    print("Accuracy: ", accuracy_score(y_true, y_pred_eer))
    print('\n===========================================')
    return eer, eer_th
    """
