import numpy as np
from scipy import signal

def average_sigs(sigs, n=5):
    # Average 5 indiv sigs 
    ret = []
    i = 0
    while i + 5 < len(sigs):
        ret.append(np.mean(sigs[i:i+5], axis=0))
        i += 1
    return ret

"""
1. first filter the raw signal with a third order Savitzky-Golay filter with 0.01s time span
2. Remove baseine wandering through detrending (matlab detrend, which I think has a scipy version)
3. To segment, use Pan-Tompkins to find ECG Rpeaks. Then locate the AO peak in each SCG by finding the maximum peak that is nearest to the corresponding ECG R-peak. 
4. Then segment into 1sec sections by 0.5sec before the AO peak and 0.5 after it
"""
def filter(scg, orig_fs):
    window_length = int(0.01*orig_fs) #window_length needs be > polyorder
    if window_length < 4:  window_length = 5  # for really small fs
    scg = signal.savgol_filter(scg, window_length=window_length, polyorder=3)
    scg = signal.detrend(scg)
    return scg

def get_AO_peaks(scg, ecg_rpeaks, fs):
    # get the max peak that is closest to the corresponding ecg r-peak? okay idk really but heuristic I think (within 02ms from the other paper)
    """
    ao_peaks = []
    # generate tuples of the rpeaks (peak and then the next one)
    rpeak_windows = zip(ecg_rpeaks[:-1], ecg_rpeaks[1:])
    for r0, r1 in rpeak_windows:
        # try this: they say 'nearest' but dont say what that means
        # r1 = int(r0+((r1-r0)*0.67)) # two thirds of the way, nope, not helpful
        # Get location of the highest peak in this cycle and assume it's AO 
        a = np.argmax(scg[r0:r1])
        a = a+r0
        ao_peaks.append(a)
    return ao_peaks
    """
    return ecg_rpeaks  # I mean it's pretty close 
    
def segment(scg, ao_peaks, fs, time_window=1.0):
    num_before = int(0.5*time_window*fs)
    num_after = int(time_window*fs) - num_before
    sigs = []
    for a in ao_peaks:
        if a-num_before < 0:  continue
        if a+num_after > scg.size:  continue
        seg = scg[a-num_before:a+num_after]
        sigs.append(seg)
    return sigs




def min_max_norm(segments):
    # norm 0 to 1 
    ret = []
    for s in segments:
        mi, ma = np.min(s), np.max(s)
        s = (s - mi)/(ma - mi)
        ret.append(s)
    return ret

def zscore(segments):
    ret = []
    for s in segments:
        mu, sig = np.mean(s), np.std(s)
        s = (s - mu)/sig
        ret.append(s)
    return ret

