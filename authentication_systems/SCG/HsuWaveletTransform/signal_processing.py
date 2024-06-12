import numpy as np
import pywt

# for other wavelets:
from ssqueezepy import Wavelet, cwt
from ssqueezepy.experimental import scale_to_freq

from scipy import signal
from scipy import ndimage

"""
1. first filter the raw signal with a third order Savitzky-Golay filter with 0.01s time span
2. Remove baseine wandering through detrending (matlab detrend, which I think has a scipy version)
3. To segment, use Pan-Tompkins to find ECG Rpeaks. Then locate the AO peak in each SCG by finding the maximum peak that is nearest to the corresponding ECG R-peak. 
4. Then segment into 1sec sections by 0.5sec before the AO peak and 0.5 after it
"""
def filter(scg, orig_fs, window_length = 0.01):
    window_length = int(window_length*orig_fs) #window_length needs be > polyorder
    if window_length < 4:  window_length = 5  # for really small fs
    scg = signal.savgol_filter(scg, window_length=window_length, polyorder=3)
    scg = signal.detrend(scg)
    return scg

def detrend_each(scg):
    # My SCG data is pretty erratic -- need to make it the same scale (either doing this or zscoring, but don't want to mess with the data) first
    sigs = []
    for s in scg:
        sigs.append(signal.detrend(s))
    return sigs

def get_AO_peaks(scg, ecg_rpeaks, fs):
    # get the max peak that is closest to the corresponding ecg r-peak? okay idk really but heuristic I think (within 02ms from the other paper)
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



"""
Wavelet transform 
"acquire a time-frequency patterm for the matching"
1. Continuous wavelet transform on each 1 sec of SCG data. They looked at morse, gabor, bump. note that pywavelets have: 
    gabor are very similar to analytic morlet, so might just do that
    bump exists in matlab but not pywt
    morse? get it from https://github.com/OverLordGoldDragon/ssqueezepy if really needed, but it looks like gabor will be fine

"""
def wavelet_transform(sigs, type='gabor', fs=128):
    transformed = []
    sampling_period =1/fs
    scales = np.linspace(0.1, 1, num=64)  # approx what matlab does
    #scales = np.logspace(-0.1, 1, 50)
    #scales = np.linspace(0.1, 3, num=32)
    if type == 'gabor': 
        for sig in sigs:
            wavelet = pywt.ContinuousWavelet('morl')  # equivalent in pwt is "morlet"
            # wavelet should be X(a,b) where a=sscaling factor, b=translation value
            # f = pywt.scale2frequency(wavelet, 1)/fs  # for debugging
            #scales = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
            #scales = np.load('scales.npy')  # default matlab ones for this signa
            coeffs, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=sampling_period)
            #twoD_transform.append(np.stack([coeffs, freqs]))
            transformed.append(coeffs)  # 64x128
    elif type == 'morse': # use ssqueezepy because it's not iplemented in pywt
        for sig in sigs:
            wavelet = Wavelet('gmw')
            # scale='log-piecewise'
            coeffs, _ = cwt(sig, wavelet, scales, fs=fs)
            #transformed.append(Wx)
            if np.sum(np.isnan(coeffs)) > 0:  import pdb; pdb.set_trace()
            transformed.append(np.abs(coeffs))
    elif type == 'bump':
        freq_interval = [4.4, 5.6]  # hz
        raise NotImplementedError
        
    else:
        raise ValueError

    return np.array(transformed)  # num-sigs x 64 x 128



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


# matching: ssim, implemented exactly from equation in paper

def ssim(feat1, feat2):
    L = feat1.max() - feat1.min()  # range of the signals
    L = max(L, feat2.max() - feat2.min())

    mux, muy = feat1.mean(), feat2.mean()
    sigx, sigy = feat1.var(), feat2.var()
    sigxy = np.cov(feat1, feat2)[0][1]

    c1 = (0.01*L)**2
    c2 = (0.03*L)**2

    numerator = (2*mux*muy + c1) * (2*sigxy + c2)
    denominator = (mux**2 + muy**2 + c1) * (sigx**2 + sigy**2 + c2)

    return numerator / denominator

