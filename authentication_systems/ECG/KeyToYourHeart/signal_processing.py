from biosppy.signals import tools as tools
import neurokit2 as nk 
import numpy as np
import scipy.signal as scipy_signal
from math import ceil, floor
from custom_datasets import processing
class SignalProcessor:
    """
    1. Signal Filtering
    2. Segmentation
    3. Outlier Removal
    4. Feature Selection
    5. Tempalte Matchin

    Following filter details here https://groups.inf.ed.ac.uk/tulips/projects/1718/samarin.pdf
    """

    def __init__(self, fs):
        self.fs = fs

    def process(self, signal):
        signal = processing.zscore(signal)
        signal = self.bandpass_filter(signal)
        signal = self.segmentation(signal)
        signal = self.resample(signal)
        signal = self.remove_outliers(signal)
        signal = self.min_max_norm(signal)  # make sure everything on same scale
        return signal

    def min_max_norm(signal):
        pass

    def bandpass_filter(self, signal):
        # signal should be 1 x n
        sampling_rate = self.fs
        order = 3
        order = int(0.3*sampling_rate)
        filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR', #ftype='butter',
                                  band='bandpass',
                                  order=order,
                                  frequency=[1, 20],
                                  sampling_rate=sampling_rate)

        return filtered

    def segmentation(self, signal):
        # Segmented around peaks:

        _, r_peaks = nk.ecg_peaks(signal[0], sampling_rate=self.fs)
        r_peaks = r_peaks['ECG_R_Peaks'].tolist()

        # Now segment 
        r_peak_diffs = np.diff(r_peaks)
        segments = [signal[:, r - floor(0.3*rdiff): r + floor(0.67*rdiff)] for r, rdiff in zip(r_peaks[1:], r_peak_diffs)]

        return segments

    def resample(self, segments):
        # "Each segment was resampled to contain 250 samples"
        num_samp_target = 250 
        arr = []
        for seg in segments:
            resized = scipy_signal.resample(seg, num_samp_target, axis=1)
            arr.append(resized)
        return np.concatenate(arr)


    def remove_outliers(self, segments):
        
        median_waveform = np.median(segments, axis=0)
        
        distances = [np.linalg.norm(seg - median_waveform) for seg in segments]
        sorted_distances = np.argsort(distances)

        kept_segments = [segments[i] for i in sorted_distances[:ceil(0.8*len(distances))]]

        return kept_segments

    def min_max_norm(self, segments):
        # Apparently he zscores each segment
        new_segments = []
        for seg in segments:
            mi, ma = seg.min(), seg.max()
            #mean = seg.mean()
            #std = seg.std()
            #seg -= mean
            #seg /= std
            news = (seg - mi)/(ma - mi)
            new_segments.append(news)
        return new_segments

        
