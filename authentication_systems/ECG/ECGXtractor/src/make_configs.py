"""
Helper to write the configs for given dataset
Assumes file structure as created by prepare_dataset.py


Two modes: 'autoencoder' and (default) siamese. 
If autoencoder, there's only one config for training (training does train-val) to train the autoencoder. Uses the train data, which is (0.5, 0.75) and has train/val 
Else,
    train: for training siamese. Uses the train data, which is (0.5, 0.75) and has train/val 
    test: for evaluating auth. Uses the test data, which is (0.75, 1.0) and only has val 


Does train and test
"""

import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Make training config")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--spoof_name", type=str, default='original', help='If provided, does that instead of original')
    parser.add_argument('--autoencoder_path', type=str, default=None, help='hd5 path for trained autoencoder')
    parser.add_argument("--autoencoder", action='store_true')
    parser.add_argument('--eval', action='store_true', help='Makes config for only a test split, made using prepare_dataset --eval')
    parser.add_argument('--name', type=str, default='', help="append an optional name to the fn of the config")
    args = parser.parse_args()

    assert not (args.autoencoder and args.autoencoder_path), "Either make an autoencoder ocnfig or make a regular config with a given ae path"

    template_dir = 'configs'
    dataset = args.dataset 
    spoof_name = args.spoof_name
    extra_name = '_'+args.name if args.name else ''

    if args.autoencoder:
        template_dir = os.path.join(template_dir, 'autoencoder')
        template = os.path.join(template_dir, 'template_config_autoencoder.json')
        mode = "autoencoder"
    else:
        template_dir = os.path.join(template_dir, 'siamese')
        if args.eval:
            template = os.path.join(template_dir, 'template_config_test.json')
            mode = "test"
        else:
            template = os.path.join(template_dir, 'template_config_train.json')
            mode = "train"

    # configs/<autoencoder|siamese>/<bidmc>/
    save_dir = os.path.join(template_dir, dataset)
    os.makedirs(save_dir, exist_ok=True)


    def fixnames(old):
        return old.replace('datasetname', dataset).replace('spoofname', spoof_name)



    ## Training:
    if mode == "train":
        # Train siamese verification network 

        with open(template) as fp:  # from "settings/template_config_train.json"
            template = json.load(fp)

        # Change the experiment name
        template['experiment'] = fixnames(template['experiment'])

        # Data path (this is the dir where the ds files were created using prepare_dataset.py) 
        template['data_path'] = [fixnames(template['data_path'][0])]

        # Split csv paths (these are the csvs that were created using prepare_dataset.py)
        template['train'] = fixnames(template['train'])
        template['val'] = fixnames(template['val'])
        template['test'] = fixnames(template['test'])

        template['base_save_dir'] = fixnames(template['base_save_dir'])
        template['csv_train_path'] = fixnames(template['csv_train_path'])

        # Optionally provide new autoencoder
        if args.autoencoder_path is not None:  template['extractor_path'] = args.autoencoder_path


        # Write out
        fn = os.path.join(save_dir, f'config_train{extra_name}.json')
        with open(fn, 'w') as fp:
            print("Writing: ", fn)
            json.dump(template, fp, indent=2)


    # Now testing:

    elif mode == "test":

        with open(template) as fp:
            template = json.load(fp)

        template['experiment'] = fixnames(template['experiment'])
        template['data_path'] = [fixnames(template['data_path'][0])]
        template['test'] = fixnames(template['test'])
        
        if args.autoencoder_path is not None:  template['extractor_path'] = args.autoencoder_path

        fn = os.path.join(save_dir, f"config_test_{spoof_name}{extra_name}.json")
        with open(fn, 'w') as fp:
            print("Writing: ", fn)
            template = json.dump(template, fp, indent=2)
    

    elif mode == "autoencoder":

        with open(template) as fp:
            template = json.load(fp)

        template['experiment'] = fixnames(template['experiment'])
        template['data_path'] = [fixnames(template['data_path'][0])]
        template['train'] = fixnames(template['train'])
        template['val'] = fixnames(template['val'])
        template['test'] = fixnames(template['test'])

        template['base_save_dir'] = fixnames(template['base_save_dir'])

        fn = os.path.join(save_dir, f'config_autoencoder.json')
        with open(fn, 'w') as fp:
            print("Writing: ", fn)
            json.dump(template, fp, indent=2)

