"""
Data processing to extract the m most discriminative QRS complexes from the trace. 
"""
from biosppy.signals import tools
from biosppy.signals.ecg import ecg as biosppy_ecg
import scipy.signal as scipy_signal
import numpy as np
from itertools import cycle
from scipy.signal import detrend

class SignalProcessor:

    MAX_N = 500  # max QRS complexes to look at to start

    def __init__(self, fs=128, m=8, resample_from=None):
        """
        win_len: winow_len to segment the data before reducing (seconds)
        m: number of QRS complexes in final template vector V (input to CNN)
        resample_from: int, the fs to expect the ecg in self.process to come in as. Resample first

        Note: after this, the ecg wlil be segmented into samples of 10sec and mean-normalized
        """
        self.fs = fs
        self.m = m

        resample = resample_from is not None and resample_from != fs
        self.resample_from = resample_from if resample else None # if resample, then first resmples the given ecg to self.fs
        
    def process(self, ecg):
        """ All steps:
            1. Resample
            2. Filter
            3. Peak detection 
            4. Segment into indiv heartbeats
            5. Filter out anomolous heartbeats 
            6. Min-max norm 
        """
        if self.resample_from is not None:
            ecg = self.resample(ecg, orig_fs=self.resample_from, target_fs=self.fs)
        ecg = self.filter(ecg)

        detrend(ecg, overwrite_data=True)
        ecg, r_peaks = self.get_R_points(ecg)
        if ecg is None:  return None # snippet too small
        ecg = self.segment(ecg, r_peaks)
        ecg = self.reduce(ecg)
        if ecg is None:  return None # why
        ecg = self.min_max_norm(ecg)
        return ecg

    def min_max_norm(self, sig):
        ' From cardiogan paper: [-1, 1] normalize each windowed segment. Done per lead. Not what DeepECG does, but makes no sense without it'
        r = [-1, 1]  # new range
        minx = sig.min()
        maxx = sig.max()
        newl = r[0] + ( (sig - minx)*(r[1] - r[0]) ) / (maxx - minx)
        return newl

    def resample(self, ecg, orig_fs, target_fs):
        num_samp_orig = len(ecg)
        num_samp_target = round( (target_fs * num_samp_orig) / orig_fs )
        resized = scipy_signal.resample(ecg, num_samp_target)
        return resized

    def filter(self, ecg):
        """
        First we reduce the noise by applying a notch IIR filter and normalize the signal's
        baseline by using a third order high-pass Butterworth filter with cutoff 
        frequency of 0.5Hz
        """
        filtered, _, _ = tools.filter_signal(signal = ecg,
            ftype='butter',
            band='highpass',
            order=3,
            frequency=0.5,
            sampling_rate=self.fs)

        return filtered

    def get_R_points(self, ecg):
        """
        Second, we estimate the position of the n R fiducial points by using an automatic
        labeling tool (Vision Premier).
        """
        try:
            res = biosppy_ecg(ecg, sampling_rate=self.fs, show=False)
            r_peaks = res[2]  # r peak locations
            return ecg, r_peaks
        except:
            # Not enough peaks 
            return None, None

    def segment(self, ecg, r_peaks):
        """
        After that, we select a time window of 0.125s for each R point, obtaining a 
        vector H of n QRS complexes
        """
        seg_len = round(0.125 * self.fs)  # 0.125seconds in num points 
        H = [] 
        for rix, r in enumerate(r_peaks[1:-1]):  # forget the first and last
            if rix > self.MAX_N:  break
            seg = ecg[r-seg_len: r+seg_len]
            H.append(seg)
        return H

    def reduce(self, complexes):
        """
        We extract the m most discriminative complexes and stitch them together, obtaining
        feature vector V
        Note: to estimate the quality of a QRS complex, we calculate its correlation with
        the average QRS pattern of H. Then <etc just read the paper>
        """
        if len(complexes) == 0:
            print("Sig is too short, didnt find enough rpeaks")
            return None
        if len(complexes) < self.m:
            # Not enough, take all of them and repeat
            num_extra_needed = self.m - len(complexes)
            gen_extra = cycle(complexes)
            while len(complexes) < self.m:
                complexes.append(next(gen_extra))
            #extra_complexes = [complexes[i] for i in range(num_extra_needed)]
            #complexes.extend(extra_complexes)
            ecg = np.array(complexes).flatten()
            return ecg

        average_H = np.mean(complexes, axis=0)
        corrs = []
        for c in complexes:
            corrs.append(np.corrcoef(c, average_H)[0][1])
        # Then select the top self.m
        sort_ix = np.argsort(corrs)
        reduced = [complexes[i] for i in sort_ix[:self.m]]
        # Then stitch together
        ecg = np.array(reduced).flatten()
        return ecg

