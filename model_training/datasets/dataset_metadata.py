from dataclasses import dataclass

@dataclass
class DualDatasetMetadata:
    """
    This is required to be set as 'METADATA' on each kind of *Dataset. 
    Contains info that is used in plotting, eval, debugging (like "A" datatype, FS, etc...) that should be passed
        around with the Dataset
    """
    NAME_A: str 
    NAME_B: str 
    NUM_LEADS_A: int 
    NUM_LEADS_B: int 
    FS_A: int 
    FS_B: int 
    SAMPLE_LEN: int 
    NUM_SAMPLES_TRAIN: int 
    NUM_SAMPLES_TEST: int 

    def __post_init__(self):
        # Over-engineering, why not
        assert self.NAME_A in ['ECG', 'EGM', 'PPG', 'SCG', 'BCG', 'RR', 'ACC', 'RPPG']
        assert self.NAME_B in ['ECG', 'EGM', 'PPG', 'SCG', 'BCG', 'RR', 'ACC', 'RPPG']

