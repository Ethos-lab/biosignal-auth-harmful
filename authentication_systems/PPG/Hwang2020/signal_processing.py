import numpy as np
from custom_datasets.processing import filter_signal
import pywt  # for the wavelet functio n 
import dtw  # for dtw
import scipy
import neurokit2 as nk2
from biosppy.signals import tools as tools
from scipy import signal

def min_max_norm(ppg, rang=[-1,1]):
    nl = ppg.shape[0]
    for l in range(nl):
        minx = ppg[l].min()
        maxx = ppg[l].max()
        newl = rang[0] + ( (ppg[l] - minx)*(rang[1] - rang[0]) ) / (maxx - minx)
        ppg[l] = newl
    return ppg
    

def filter(ppg, fs):
    " 4th order butterworth [0.5, 18] "
    filtered, _, _ = tools.filter_signal(signal=ppg,
                                        ftype='butter',
                                        band='bandpass',
                                        order=4,
                                        frequency=[0.5, 18],
                                        sampling_rate=float(fs))
    return filtered

def detrend_each(ppgs):
    # My SCG data is pretty erratic -- need to make it the same scale (either doing this or zscoring, but don't want to mess with the data) first
    sigs = []
    for p in ppgs:
        sigs.append(signal.detrend(p))
    return sigs
    
def segmentation(ppg, fs):
    # biosppy for ppg systolic peak detection 
    # Returns two types of segments: the first from start-syspeak-end for DWT, and
    #   the second from syspeak-end-start-syspeak (shifted), for the ZT (nope not anymore)
    # UPDATE: uses nk2 to actually get the syspeaks, (elgendi)

    ppg = ppg[0]
    _, info = nk2.ppg_process(ppg, sampling_rate=fs)
    syspeaks = info['PPG_Peaks']

    ipis = np.diff(syspeaks)
    average_ipi = np.median(ipis)  
    upper_limit = 1.2*average_ipi
    lower_limit = 0.4*average_ipi
    numsamps_before = int(0.2*fs)
    start_points = []
    filtered_peaks = []
    for peak, ipi in zip(syspeaks, ipis+[-1]):
        if ipi == -1:  # this is the last one
            break
        if ipi > upper_limit or ipi < lower_limit:  
            start_points.append(-1)  # we wont know if we should have excluded the last one we added until we've processed this one... so just append -1 as a placeholder           
            filtered_peaks.append(-1)
        else:
            sec = ppg[peak-numsamps_before:peak-1]
            minsec = np.argmin(sec)
            start_points.append(minsec+peak-numsamps_before) # offset
            filtered_peaks.append(peak)

    # Now segment using the start points:
    segments1 = []
    for sp0, sp1 in zip(start_points[:-1], start_points[1:]):
        if sp0 == -1 or sp1 == -1:  continue  
        if sp0 == sp1:  continue  
        section = ppg[sp0:sp1]
        segments1.append(section)

    return segments1  #, (segments2, startpoints_filtered)

def min_max_scale(seg):
    if seg.size == 0: import pdb; pdb.set_trace()
    lower, upper = seg.min(), seg.max()
    orig_range = upper - lower
    seg = seg - lower
    seg = seg/orig_range
    return seg

def zscore(ppg):
    mu, sig = ppg.mean(), ppg.std()
    ppg = ppg - mu
    ppg = ppg / sig
    return ppg


def shift_segments(segments, final_len, pad=True):
    # Convenience method for ZT and IN: 
    shifted = []
    for seg1, seg2 in zip(segments[:-1], segments[1:]):
        # first shift so that it's peak to next peak instead of trough to trough
        firsthalf = seg1[np.argmax(seg1):]  # peak of first segment
        secondhalf = seg2[:np.argmax(seg2)]  # peak of second segment
        orig_sizes = firsthalf.size, secondhalf.size
        if not pad:
            seg = np.concatenate([firsthalf, secondhalf])
            seg = seg[:final_len]
        else:
            numpad_middle = final_len - (len(firsthalf) + len(secondhalf))
            if numpad_middle > 0:
                firsthalf = np.pad(firsthalf, (0, numpad_middle), mode='constant', constant_values=0)  
                seg = np.concatenate([firsthalf, secondhalf])
            elif numpad_middle < 0:  # if it's actually larger than final_len
                """
                # incredible annoying -- remove half from the end of firshalf and half from the beginning of secondhalf
                if numpad_middle % 2 == 0:  # even
                    remove_1, remove_2 = int(numpad_middle/2), int(numpad_middle/2)
                else:  # odd
                    remove_1, remove_2 = int(numpad_middle/2), int(numpad_middle/2)-1
                if remove_1 < 0:  firsthalf = firsthalf[:remove_1]
                secondhalf = secondhalf[-1*remove_2:]
                """
                seg = np.concatenate([firsthalf, secondhalf])
                seg = seg[:final_len]  # Never mind, just concat and remove from the end
            else:
                seg = np.concatenate([firsthalf, secondhalf])
        shifted.append(seg)
    return shifted

def stretching(segments, fs, stretch_type='DTW', scale=5, final_len_scale=1.25):
    " Stretches; also min_max and zscores "
    final_len = int(fs*final_len_scale) 
    stretched = []

    if stretch_type == 'DTW':
        template = pywt.Wavelet('db4')
        # Hardcode some because there's no great way to do this with wavefun
        scaling_fn, wavelet_fn, _ = template.wavefun(scale)  # for scaling=5, len is 225.
        # Resample it to final_len so it's consistent with ZT, IN etc 
        scaling_fn = scipy.signal.resample(scaling_fn, final_len+1) 
        
        for seg in segments[:-1]: # get rid of the last cause ZT doesnt have it
            # min max scale first
            seg = min_max_scale(seg)
            # DTW with DB34 template
            #alignment = dtw.dtw(x=seg, y=scaling_fn, dist=lambda x, y: (x-y)**2)
            alignment = dtw.dtw(x=seg, y=scaling_fn)
            wq = dtw.warp(alignment,index_reference=False)  # who knows if any of this is right
            warped = seg[wq]
            # zscore
            if np.sum(warped) == 0: 
                seg = warped 
            else:
                seg = zscore(warped)
            stretched.append(seg)

    elif stretch_type == 'ZT':
        segments = shift_segments(segments, final_len)
        for seg in segments:
            # Then FFT
            complexoutput = scipy.fft.fft(seg)
            mag = np.abs(complexoutput)
            mag = zscore(mag)
            stretched.append(mag)

   
    elif stretch_type == 'IN':
        # deal with this later if needed -- dont get how they are still using 
        # start and end poitns in the freq domain, makes no sense.  
        segments = shift_segments(segments, final_len, pad=False)
        for seg in segments:
            complexout = scipy.fft.fft(seg)
            mag, angle =  np.abs(complexout), np.angle(complexout)
            # Then pad somehow?  This is so bizarre, but I guess choose the middle:

            lenmag = len(mag)
            numpad = final_len - lenmag
            if numpad > 0:
                mag = np.concatenate([mag[:lenmag//2], np.zeros(numpad), mag[lenmag//2:]])
                ang = np.concatenate([angle[:lenmag//2], np.zeros(numpad), angle[lenmag//2:]])
            elif numpad == 0:
                ang = angle
            else:
                import pdb; pdb.set_trace()

            newft = mag**(np.exp(1j*ang))
            reversefft = (final_len/len(newft))*scipy.fft.ifft(newft)
            seg = np.real(reversefft)
            #seg = zscore(seg)
            #import pdb; pdb.set_trace()
            stretched.append(seg)
    

    elif stretch_type == 'ZP':
        # Just zero pad at the end dammit, this doesn't need to be difficult
        for sii, s in enumerate(segments[:-1]):
            if len(s) >= final_len:  
                news = s[:final_len]
            else:
                lastval = s[-1]
                numpad = final_len - len(s)
                padded = np.pad(s, (0, numpad), mode='constant', constant_values=lastval)
                news = padded
            news = zscore(news)
            stretched.append(news)  # NOTE I changed this

    elif stretch_type == 'VV':  # veena creation
        
        for seg1, seg2 in zip(segments[:-1], segments[1:]):
            numpad = final_len - len(seg1)
            if numpad > 0:
                s = np.concatenate([seg1, seg2[:numpad]])
                if final_len > len(s):  # stil lmore
                    remaining = final_len - len(s)
                    s = np.pad(s, (0, remaining), mode='constant', constant_values=s[-1])
            elif numpad < 0:
                s = seg1[:final_len]
            else:
                s = seg1
            s = zscore(s)
            stretched.append(s)

    return stretched



def remove_outliers(segs):
    ordered_segs = []
    distances = []
    mean_seg = np.mean(segs, 0)
    for s in segs:
        distance = (mean_seg-s).sum()  # actuall mse but it's monotonic
        distances.append(distance)
    ordered_distances = np.argsort(distances)
    ordered_segs = [segs[i] for i in ordered_distances]
    return ordered_segs 


def data_aug(segs, noise_type='combination'):
    new_segs = [] 
    if noise_type == 'gaussian':
        for s in segs:
            len_pulse = len(s)
            gauss = np.random.normal(0, 0.33, len_pulse)
            new_segs.append(s+gauss) 
    elif noise_type == 'sloping':
        for s in segs:
            len_pulse = len(s)
            uniform = np.random.uniform(-1, 1, len_pulse)
            new_s = s+(3/len_pulse)*uniform
            new_segs.append(new_s)
    elif noise_type == 'combination':
        for s in segs:
            len_pulse = len(s) 
            gauss = np.random.normal(0, 0.33, len_pulse)
            slope = np.random.uniform(-1, 1, len_pulse)
            new_s = s + gauss + ((3/len_pulse)*slope)
            new_segs.append(new_s)
    return new_segs


