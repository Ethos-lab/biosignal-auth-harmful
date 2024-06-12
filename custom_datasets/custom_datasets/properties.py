"""
Dataset properties that get accessed from ./datasets.py
"""
from dataclasses import dataclass

hdd = '/path/to/root/hdd'

def get_props(dataset, spoof, spoof_name):
    attr_name = helper_attr_name(dataset, spoof, spoof_name)
    if attr_name not in globals():
        raise NotImplementedError('Dont have this dataset in properties.py')

    return globals()[attr_name]  # this is a weird way to do this but okay

def helper_attr_name(dataset, spoof, spoof_name):
    # This is the convention I'm using... TODO might change
    dataset = dataset.split('-')[0] # For handling multiple sesssions 
    if not spoof:  return dataset
    if spoof_name == 'original':  return dataset  # okay but something went wrong
    return f"{dataset}_spoof_{spoof_name}"

@dataclass
class DatasetProps:
    NAME: str  # also the name of the instantiated var, but makes things clearer if it's here as well
    SOURCE: str  # 'original' or spoof_name; 'their_cardiogan' if their weight
    READ_PATH: str
    DATA_TYPES: list
    FS: dict
    TRAIN_SET: list
    TEST_SET: list
    OPTIONAL_ALT_READ_PATH: str = ""  # Added because the video stuff is split b/e 2 drives


bidmc = DatasetProps('bidmc',
    'original',
    hdd + 'egm-ecg-cyclegan/data/ECGPPG_BIDMC/physionet.org/files/bidmc/1.0.0/bidmc_csv',
    ['ecg', 'ppg', 'rr', 'spo2'], # TODO
    {'ecg': 125, 'ppg': 125},
    ['05', '22', '21', '31', '02', '07', '28', '50', '29', '42', '40', '44', '24', '13', '26', '01', '48', '43', '23', '45', '53', '06', '36', '10', '03', '32',  '16', '04', '25', '20', '35', '11', '08', '46'],
    ['17', '34', '12', '33', '14', '19', '27', '37', '52', '51', '18']
)

bidmc_spoof_my_cardiogan_contrastive = DatasetProps('bidmc',
    'my_cardiogan_contrastive',
    hdd + 'spoofed/my_cardiogan_contrastive/ecgppg_bidmc',
    ['ecg', 'ppg'],
    {'ecg': 128, 'ppg': 128},
    ['05', '22', '21', '31', '02', '07', '28', '50', '29', '42', '40', '44', '24', '13', '26', '01', '48', '43', '23', '45', '53', '06', '36', '10', '03', '32',  '16', '04', '25', '20', '35', '11', '08', '46'],
    ['17', '34', '12', '33', '14', '19', '27', '37', '52', '51', '18']
)
capno = DatasetProps('capno', 
    'original', 
    hdd + 'egm-ecg-cyclegan/data/ECGPPG_CAPNO/data/mat', 
    ['ecg', 'ppg', 'co2'],
    {'ecg': 300, 'ppg': 300, 'co2': 300},
    ['0009', '0015', '0016', '0018', '0023', '0028', '0029', '0030', '0031', '0032', '0035', '0038', '0103', '0104', '0105', '0115', '0121', '0122', '0123', '0125', '0127', '0128', '0133', '0134', '0142', '0147', '0148', '0149', '0150', '0309', '0311', '0312', '0313'],
    ['0322', '0325', '0328', '0329', '0330', '0331', '0332', '0333', '0370']
)

capno_spoof_my_cardiogan_contrastive = DatasetProps('capno',
    'my_cardiogan_contrastive',
    hdd + 'spoofed/my_cardiogan_contrastive/ecgppg_capno',
    ['ecg', 'ppg'],
    {'ecg': 128, 'ppg': 128},
    ['0009', '0015', '0016', '0018', '0023', '0028', '0029', '0030', '0031', '0032', '0035', '0038', '0103', '0104', '0105', '0115', '0121', '0122', '0123', '0125', '0127', '0128', '0133', '0134', '0142', '0147', '0148', '0149', '0150', '0309', '0311', '0312', '0313'],
    ['0322', '0325', '0328', '0329', '0330', '0331', '0332', '0333', '0370']
)


dalia = DatasetProps('dalia',
    'original',
    hdd + "egm-ecg-cyclegan/data/ECGPPG_DALIA/PPG_FieldStudy",
    ['ecg', 'ppg', 'acc', 'eda', 'temp'],
    {'ecg': 700, 'ppg': 64, 'acc': 32, 'eda': 4, 'temp': 4},
    ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'],
    ['S13', 'S14', 'S15']
)

dalia_spoof_my_cardiogan_contrastive = DatasetProps('dalia',
    'my_cardiogan_contrastive',
    hdd + 'spoofed/my_cardiogan_contrastive/ecgppg_dalia',
    ['ecg', 'ppg'],
    {'ecg': 128, 'ppg': 128},
    ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'],
    ['S13', 'S14', 'S15']
)

wesad = DatasetProps('wesad',
    'original',
    hdd + 'egm-ecg-cyclegan/data/ECGPPG_WESAD/WESAD',
    ['ecg', 'ppg', 'eda', 'emg', 'temp', 'rr', 'acc'], # TODO eda and temp from both devices
    {'ecg': 700, 'ppg': 64},
    ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14'],
    ['S15', 'S16', 'S17']
)

wesad_spoof_my_cardiogan_contrastive = DatasetProps('wesad',
    'my_cardiogan_contrastive',
    hdd + 'spoofed/my_cardiogan_contrastive/ecgppg_wesad',
    ['ecg', 'ppg'],
    {'ecg': 128, 'ppg': 128},
    ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14'],
    ['S15', 'S16', 'S17']
)

cardiogan4 = DatasetProps('cardiogan4',
    'original',
    hdd,
    ['ecg'],
    {'ecg': 128},
    [],
    ['ugh this sucks, why did I do it like this']
)

hcitagging = DatasetProps('hcitagging',
    'original',
    hdd + 'pyvhr/HCITagging',
    ['video', 'ecg', 'eeg'],
    {'video': 60.9708, 'ecg': 256, 'rppg': 61, 'ppg': 128},  # 12, 15, 26 are recording errors
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',  '13', '14', '16', '17', '18', '19', '20'],
    ['21', '23', '24', '25', '27', '28', '29', '30']  # and 22 just looks wrong
)

hcitagging_spoof_my_cardiogan_contrastive = DatasetProps('hcitagging',
    'my_cardiogan_contrastive',
    hdd + 'spoofed/latest/rppgecg_hcitagging',
    ['ecg'],
    {'ecg': 128},
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',  '13', '14', '16', '17', '18', '19', '20'],
    ['21', '23', '24', '25', '27', '28', '29', '30']  # and 22 just looks wrong
)
    
bedbased = DatasetProps('bedbased',
    'original',
    hdd + 'egm-ecg-cyclegan/data/ECGBCG_BEDBASED',
    ['ecg', 'bcg', 'ppg'],
    {'ecg': 1000, 'ppg': 1000, 'bcg': 1000},
    ['X0132', 'X1001', 'X1002', 'X1003', 'X1004', 'X1005', 'X1006', 'X1007', 'X1008', 'X1009', 'X1010', 'X1011', 'X1012', 'X1013', 'X1014', 'X1019', 'X1020', 'X1021', 'X1022', 'X1023', 'X1024', 'X1025', 'X1026', 'X1027', 'X1028', 'X1029', 'X1030', 'X1031', 'X1033', 'X1034'],
    ['X1035', 'X1037', 'X1038', 'X1039', 'X1040', 'X1042', 'X1043', 'X1044', 'X1046', 'X1047']
    )

bedbased_spoof_my_cardiogan_contrastive = DatasetProps('bedbased',
    'my_cardiogan_contrastive',  # from ECG
    hdd + 'spoofed/my_cardiogan_contrastive/ecgbcg_bedbased',
    ['bcg'],
    {'bcg': 128},
    [],
    ['X0132', 'X1001', 'X1002', 'X1003', 'X1004', 'X1005', 'X1006', 'X1007', 'X1008', 'X1009', 'X1010', 'X1011', 'X1012', 'X1013', 'X1014', 'X1019', 'X1020', 'X1021', 'X1022', 'X1023', 'X1024', 'X1025', 'X1026', 'X1027', 'X1028', 'X1029', 'X1030', 'X1031', 'X1033', 'X1034', 'X1035', 'X1037', 'X1038', 'X1039', 'X1040', 'X1042', 'X1043', 'X1044', 'X1046', 'X1047']
    )

bedbased_spoof_my_cardiogan_contrastive_from_ppg = DatasetProps('bedbased',
    'my_cardiogan_contrastive_from_ppg',  # from PPG to BCG
    hdd + 'spoofed/my_cardiogan_contrastive_from_ppg/ppgbcg_bedbased',
    ['bcg', 'ppg'],
    {'bcg': 128, 'ppg': 128, 'ecg': 128},
    [],
    ['X0132', 'X1001', 'X1002', 'X1003', 'X1004', 'X1005', 'X1006', 'X1007', 'X1008', 'X1009', 'X1010', 'X1011', 'X1012', 'X1013', 'X1014', 'X1019', 'X1020', 'X1021', 'X1022', 'X1023', 'X1024', 'X1025', 'X1026', 'X1027', 'X1028', 'X1029', 'X1030', 'X1031', 'X1033', 'X1034', 'X1035', 'X1037', 'X1038', 'X1039', 'X1040', 'X1042', 'X1043', 'X1044', 'X1046', 'X1047']
    )
    
bedbased_spoof_my_cardiogan_contrastive_from_ppg_ecg = DatasetProps('bedbased',
    'my_cardiogan_contrastive_from_ppg',  # from PPG to ECG but bedbased
    hdd + 'spoofed/my_cardiogan_contrastive_from_ppg_ecg/ppgecg_bedbased',
    ['bcg', 'ppg'],
    {'bcg': 128, 'ppg': 128, 'ecg': 128},
    [],
    ['X0132', 'X1001', 'X1002', 'X1003', 'X1004', 'X1005', 'X1006', 'X1007', 'X1008', 'X1009', 'X1010', 'X1011', 'X1012', 'X1013', 'X1014', 'X1019', 'X1020', 'X1021', 'X1022', 'X1023', 'X1024', 'X1025', 'X1026', 'X1027', 'X1028', 'X1029', 'X1030', 'X1031', 'X1033', 'X1034', 'X1035', 'X1037', 'X1038', 'X1039', 'X1040', 'X1042', 'X1043', 'X1044', 'X1046', 'X1047']
    )

cebsdb = DatasetProps('cebsdb',
    'original',
    hdd + 'egm-ecg-cyclegan/data/ECGSCG_CEBSDB/physionet.org/files/cebsdb/1.0.0/',
    ['ecg', 'scg'],
    {'ecg': 5000, 'scg': 5000},
    ['b001', 'b002', 'b003', 'b004', 'b005', 'b006', 'b007', 'b008', 'b009', 'b010', 'b011', 'b012', 'b013', 'b014', 'b015'],
    ['b016', 'b017', 'b018', 'b019', 'b020'],
    )

cebsdb_spoof_my_cardiogan_contrastive = DatasetProps('cebsdb',
    'my_cardiogan_contrastive',
    hdd + 'spoofed/my_cardiogan_contrastive/ecgscg_cebsdb',
    ['scg'],
    {'scg': 512},  
    ['b001', 'b002', 'b003', 'b004', 'b005', 'b006', 'b007', 'b008', 'b009', 'b010', 'b011', 'b012', 'b013', 'b014', 'b015'],
    ['b016', 'b017', 'b018', 'b019', 'b020'],
    )