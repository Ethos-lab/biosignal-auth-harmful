import numpy as np
from biosppy.signals import tools as tools
import scipy.stats as scipy_stats
import scipy.signal as scipy_signal
from math import floor


def sliding_window(data, fs, window_len=4, overlap_len=0.4): # window_len and overlap_len in seconds
    ' len should be in seconds '
    arr = []
    i = 0
    l = fs*window_len  # target num samples per window
    nl = data.shape[0]  # num leads
    ns = data.shape[1]  # num samples
    shift_len = floor((window_len - overlap_len) * fs)  # if 10% overlap, we shift 90%, or (window_len - overlap_len) seconds, times fs gives num samples to shift
    while i+l < ns:
        segment = data[:, i:i+l] # section out the window
        arr.append(segment)
        i = i + shift_len # increase the shift
    return np.stack(arr)



def resample(signal, orig_fs = 700, target_fs=128):
    ''' Reamples using scipy.signal.resample, which uses fourier interpolation. the cv2.INTER_LINEAR wasnt working for some reason. 
        Make sure input signal is shape num_leads x num_sample '''
    num_samp_orig = signal.shape[1]
    num_samp_target = round( (target_fs * num_samp_orig) / orig_fs )
    resized = scipy_signal.resample(signal, num_samp_target, axis=1)
    return resized


def filter_ecg(signal, sampling_rate, freq_band=[3,45]):
    
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=freq_band,
                                  sampling_rate=sampling_rate)
    return filtered

def filter_egm(signal, sampling_rate):
    
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    return filtered

def filter_ppg(signal, sampling_rate=128, freqs=[1, 8]):

    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4, #3
                                  frequency=freqs,
                                  sampling_rate=sampling_rate)

    return filtered


def zscore(sig):
    ' From cardiogan paper: perform person-specific zscore normalization. This should be done per lead '
    nl = sig.shape[0]  # num leads
    for l in range(nl):
        mean = sig[l].mean()
        std = sig[l].std()
        sig[l] -= mean
        sig[l] /= std
    return sig

def min_max_norm(sig):
    ' From cardiogan paper: [-1, 1] normalize each windowed segment. Done per lead '
    r = [-1, 1]  # new range
    nl = sig.shape[0]
    for l in range(nl):
        minx = sig[l].min()
        maxx = sig[l].max()
        newl = r[0] + ( (sig[l] - minx)*(r[1] - r[0]) ) / (maxx - minx)
        sig[l] = newl
    return sig
