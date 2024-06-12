# Biosignal Authentication Considered Harmful Today 
Codebase to accompany [Biosignal Authentication Considered Harmful Today](): 
> Veena Krish, Nicola Paoletti, Milad Kazemi, Scott Smolka, Amir Rahmati (2024). Biosignal Authentication Considered Harmful Today. In USENIX Security Symposium (USENIX Sec). 


## Overview
This codebase is divided into three parts:  
    1. `custom_datasets`: loading and processing raw data, used across the codebase  
    2. `model_training`: training the cyclegan network used to generate spoofed biosignals  
    3. `authentication_systems`, an implementation of the systems tested in the paper   


### 1. custom_datasets
Data handlers for loading, processing and splitting data are packaged as a python module that can be imported in other sections of the overall codebase. This can be 

#### Packaging: 

First build the custom_datasets package:
```
cd custom_datasets
python -m build
```

Then install the package in the main venv as:
```
pip install --editable .
```

Notes: 
  * Need to upgrade pip beyond 21.3 in order to build and install editable
  * [PyPA User Guide](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#install-build) if you encounter any issues


#### Main usage: 

The file `datasets.py` contains generators for yielding a data sample for a given configuration, defined in `properties.py``. Arguments `datasets.get_data`:  
  * dataset_name: as defined in properties.py. E.g. 'dalia', 'capno'
  * data_type: biosignal modality. E.g. 'ECG', 'PPG'
  * spoof: bool for requesting generated/spoofed data or original. False is overriden if spoof_name supplied
  * spoof_name: name of the spoof generation method (e.g. 'cardiogan_contrastive', 'video_ecg_contrastive')
  * split: oneof: ['train', 'test', 'all'], denotes the split on subjects (note: not on time), as defined in properties.py. Generally used for training/testing the cyclegan's generalization ability
  * fraction_time: fraction of total signal time to yield, typically used for training v testing the authentication systems.  


```
from custom_datasets import datasets
generator = dataasets.get_data(dataset_name='dalia', data_type='PPG', spoof=False, spoof_name=None, split='train', fraction_time=(0, 0.5), session=None)
subject_ix, subject_name, session_name, ppg_npy = next(generator)

# For paired data (for evaluating model training):
generator = dataasets.get_data('dalia', ['PPG', 'ECG'] spoof=False, split='train')
_, _, _, paired_data = next(generator)
ppg_npy = paried_data['PPG']
ecg_npy = paried_data['ECG']
```

### model_training

Contains pytorch-based model training scripts. View example usage with `python train_contrastive_cardiogan.py --help`. DataLoaders pull paried raw data from custom_datasets.get_paired_data and shuffle to train unpaired translation. Each DataLoader specifies the source and target datatype for the given dataset. 

Example usage: 
```
# Train the main ppg -> ecg spoofing model
python train_contrastive_cardiogan.py --dataset ecgppg_cardiogan

# Train an example video -> ecg spoofing model
python train_contrastive_cardiogan.py --dataset rppgecg_hcitagging
```


### authentication_systems
The following authentication systems are implemented (using public codebases if available). Evaluation scripts are included in each directory for testing the false acceptance rate of spoofed data. 

#### ECG

##### ECGXtractor
Published as: [ECG Biometric Recognition: Review, System
Proposal, and Benchmark Evaluation. P Melzi, R Tolosana, R Vera-Rodriguez - IEEE Access, 2023.]((https://arxiv.org/pdf/2204.03992))

Implementation is obtained directly from authors' [provided codebase](https://github.com/BiDAlab/ECGXtractor) and modified to work with our datasets.

Usage: 
```
# Prepare data for <dataset_name>
python src/prepare_dataset.py --dataset <dataset_name>  # note: this creates train/train.json and train/val.json
python src/prepare_dataset.py --dataset <dataset_name> --eval  # note: this creates eval/train.json that has nothing and eval/val.json
python src/prepare_dataset.py --dataset bidmc --eval --spoof_name cardiogan_contrastive

# Make config files for autoencoder and siamese network training and testing
python src/make_configs.py --dataset <dataset_name> --autoencoder
python src/make_configs.py --dataset <dataset_name>
python src/make_configs.py --dataset <dataset_name> --eval
python src/make_configs.py --dataset <dataset_name> --spoof_name cardiogan_contrastive --eval

# Train
python src/train.py --config_file configs/autoencoder/<dataset_name>/config_autoencoder.json --rename latest
python src/train.py --config_file configs/siamese/<dataset_name>/config_train.json --rename latest

# Then predict and save the working EER, EER_threshold for a withheld validation section
python src/predict.py --dataset <dataset_name> --config_file configs/siamese/<dataset_name>/config_train.json --model_name latest --save_stats

# Eval using the predicted EER/thresh on a final withheld section, over 10 attempts
python src/eval.py --dataset <dataset_name> --model_name latest --spoof_name original
python src/eval.py --dataset <dataset_name> --model_name latest --spoof_name cardiogan_contrastive

```

##### DeepECG
Published as: [Deep-ECG: Convolutional neural networks for ECG biometric recognition. RD Labati, E Muñoz, V Piuri, R Sassi, F Scotti - Pattern Recognition Letters, 2019](https://doi.org/10.1016/j.patrec.2018.03.028)

Usage:
```
# Train
python train.py --dataset <dataset_name> --save latest 

# Test and get the EER ("--save_stats" will save the eer threshold to the model file)
python test.py --dataset <dataset_name> --model_path saved_models/<dataset_name>/latest.pt --save_stats
# Eval using the saved EER over 10 attempts on a separate section of data, on original and spoofed data

python eval.py --dataset <dataset_name> --model_path saved_models/<dataset_name>/latest.pt --spoof_name original
python eval.py --dataset <dataset_name> --model_path saved_models/<dataset_name>/latest.pt --spoof_name cardiogan_contrastive
```

##### EDITH

Published as: [Ibtehaz, Nabil, et al. [EDITH: ECG biometrics aided by deep learning for reliable individual authentication. IEEE Transactions on Emerging Topics in Computational Intelligence, 2021](Ibtehaz, Nabil, et al. "EDITH: ECG biometrics aided by deep learning for reliable individual authentication." IEEE Transactions on Emerging Topics in Computational Intelligence 6.4 (2021): 928-940.
)


Usage:
```
# Train feature extractor and siamese models (n.b. train_siamese also gets EER on a withheld set and saves to model)
python train_baseclassifier.py --dataset <dataset_name> --save base
python train_siamese.py --dataset <dataset_name> --saved_model saved_models/<dataset_name>/base.pt --save siamese

# Eval on original/spoofed datasets
python eval.py --dataset <dataset_name> --saved_base_model saved_models/<dataset_name>/base.pt --saved_siamese_model saved_models/<dataset_name>/siamese.pt --spoof_name cardiogan_contrastive
```


##### KeyToYourHeart

Published as: [A Key to Your Heart: Biometric Authentication Based on ECG Signals. N Samarin - School of Informatics, University of Edinburgh, 2018](https://arxiv.org/abs/1906.09181)

Usage: 
```
python generate_data.py --dataset <dataset_name> --spoof_name original --split train
python generate_data.py --dataset <dataset_name> --spoof_name original --split test
python generate_data.py --dataset <dataset_name> --spoof_name cardiogan_contrastive --split test

2. Train, get test EER and save
python train.py --dataset <dataset_name> --save

3. Eval on 10 trials
python eval.py --dataset <dataset_name> --model_dir saved_models/<dataset_name> --spoof_name original
python eval.py --dataset <dataset_name> --model_dir saved_models/<dataset_name> --spoof_name cardiogan_contrastive

```

#### PPG

##### CorNET

Published as: [CorNET: Deep learning framework for PPG-based heart rate estimation and biometric identification in ambulant environment. D Biswas et al. IEEE Transactions on Biomedical Circuits and Systems, 2019](https://ieeexplore.ieee.org/document/8607019)


Usage:

```
# Train and save test-split EER: 
`python train.py --dataset <dataset_name> --save latest
# Eval on orig and spoofed datasets
python eval.py --dataset dalia --models_dir saved_models/<dataset_name>/latest/
python eval.py --dataset dalia --models_dir saved_models/<dataset_name>/latest/ --spoof_name cardiogan_contrastive
```

##### Hwang2020

Published as: [Evaluation of the time stability and uniqueness in PPG-based biometric system
DY Hwang et al. IEEE Transactions on Information Forensics and Security, 2020](https://ieeexplore.ieee.org/document/9130730)

Generally based off [unofficial implementation](https://github.com/eoduself/PPG-Verification-System) (written by an author of the paper but not linked within)

Usage: 
```
# Train and get test-split EER: 
python train.py --dataset <dataset_name> --lstm --save latest
# Eval on original and spoofed sets :
python eval.py --dataset <dataset_name> --models_dir saved_models/<dataset_name>/latest/
python eval.py --dataset <dataset_name> --spoof_name cardiogan_contrastive --models_dir saved_models/<dataset_name>/latest
```

#### SCG

##### WaveletTransform

Published as: [Exploring seismocardiogram biometrics with wavelet transform.PY Hsu, PH Hsu, HL Liu.   2020 25th International Conference on Pattern Recognition (ICPR), 2021](https://ieeexplore.ieee.org/document/9412582)

Usage:
```
python train_and_eval.py --wavelet morse --spoof_name original`
python train_and_eval.py --wavelet morse --spoof_name cardiogan_contrastive`
```

##### MotionArtifactResilient

Published as: [Motion artifact resilient SCG-based biometric authentication using machine learning
PY Hsu, PH Hsu, TH Lee, HL Liu, 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) 2021](https://pubmed.ncbi.nlm.nih.gov/34891258/)

Usage:
```
python train.py --save latest
python eval.py --model_path saved_models/latest --spoof_name cardiogan_contrastive
```

#### BCG

##### Herbert2018

Published as: [Ballistocardiogram-based authentication using convolutional neural networks. Hebert, Joshua, et al. Worcester Polytechnic Institute, 2018](https://arxiv.org/abs/1807.03216)

```
python train.py --dataset <dataset_name> --save latest
python eval.py --dataset dataset --models_dir saved_models/<dataset_name>/latest
python eval.py --dataset dataset --models_dir saved_models/<dataset_name>/latest --spoof_name cardiogan_contrastive
```
##### ZhangRNN

Published as: [Ballistocardiogram based person identification and authentication using recurrent neural networks. Zhang, Xianwen, et al. 2018 11th International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI), 2018](https://ieeexplore.ieee.org/document/8633102)

```
python train.py --dataset <dataset_name> --save latest
python eval.py --dataset dataset --models_dir saved_models/<dataset_name>/latest
python eval.py --dataset dataset --models_dir saved_models/<dataset_name>/latest --spoof_name cardiogan_contrastive
```