import numpy as np
from .signal import (
    trim_signal, band_pass_filter, 
    standardization_signal, 
    remove_mv, remove_rms,
    k_under_interpolate   )
from .configs import EEG, EOG, EMG, RESP, SPO2

def prep_signal(signal, 
    lf=1.0, hf=49.9,sfreq=100 ,norm =5, pre_trim=True, post_trim=True):

    # Normalize all the other channels by removing the mean and the rms in an 18 minute rolling window, using fftconvolve for computational efficiency
    # 18 minute window is used because because baseline breathing is established in 2 minute window according to AASM standards.
    # Normalizing over 18 minutes ensure a 90% overlap between the beginning and end of the baseline window
    kernel_size = (sfreq*18*60)+1

    if pre_trim: signal.signal = trim_signal(
        signal.signal, signal.physical_min, signal.physical_max)

    # resample to sfreq Hz
    signal = signal.resample(sfreq)

    # Apply antialiasing FIR filter to each channel
    signal.signal = band_pass_filter(signal.signal, lf=lf, hf=hf, fs=signal.sample_rate)
    
    # Remove DC bias and scale for FFT convolution
    signal.signal = standardization_signal(signal.signal)

    # Compute and remove moving average with FFT convolution
    signal.signal = remove_mv(signal.signal, kernel_size)

    # Compute and remove the rms with FFT convolution of squared signal
    signal.signal = remove_rms(signal.signal, kernel_size)

    if post_trim: signal.signal = trim_signal(signal.signal, -norm, norm)

    # Scale -1 ~ 1
    signal.signal = signal.signal/norm

    return signal

def prep_spo2(signal, sfreq=100):
    
    signal.signal = k_under_interpolate(signal.signal, 30)

    # Resampling without FIR
    def gcd(x, y): # Greatest Common Divisor
        while(y): x, y = y, x % y
        return x

    def lcm(x, y): # Least Common Multiple
        return (x*y)//gcd(x,y)

    # signal.resample(sfreq)
    if sfreq%signal.sample_rate !=0:
        up   = lcm(signal.sample_rate, sfreq)//signal.sample_rate
        down = lcm(signal.sample_rate, sfreq)//sfreq
    else: 
        if signal.sample_rate > sfreq:
            up, down = 1, signal.sample_rate//sfreq
        else: 
            up, down = sfreq//signal.sample_rate, 1

    signal.signal = np.repeat(signal.signal, up)
    signal.signal = signal.signal[::down]
    signal.sample_rate=sfreq
    
    
    # Scale -0.5 ~ 0.5
    signal.signal = (signal.signal/100)-0.5
    
    return signal

def prep_edf(edf,sfreq=50):

    for idx, ch in enumerate(edf.ch_names):

        if   ch in EEG:
            hf = min([30.0, (sfreq/2)-0.01])
            edf.signals[idx] = prep_signal(edf[ch], 
            lf=2.0, hf=hf,sfreq=sfreq ,pre_trim=True, post_trim=True)

        elif ch in EOG:
            hf = min([3.0, (sfreq/2)-0.01])
            edf.signals[idx] = prep_signal(edf[ch], 
            lf=0.1, hf=3.0,sfreq=sfreq ,pre_trim=True, post_trim=False)
            
        elif ch in EMG:
            edf.signals[idx] = prep_signal(edf[ch], 
            lf=4.0, hf=(sfreq/2)-00.1,sfreq=sfreq ,pre_trim=False, post_trim=True)

        elif ch in RESP:
            hf = min([1.0, (sfreq/2)-0.01])
            edf.signals[idx] = prep_signal(edf[ch], 
            lf=0.1, hf=1.0,sfreq=sfreq, norm=1,pre_trim=False, post_trim=True)

        elif ch in SPO2:
            edf.signals[idx] = prep_spo2(edf[ch], sfreq=sfreq)

    return edf

