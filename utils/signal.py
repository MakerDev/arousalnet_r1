import numpy as np 
from mne.filter import filter_data
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

def band_pass_filter(x, lf, hf, fs=100, tb='auto'):

    return filter_data(
        x, sfreq=fs, l_freq=lf, h_freq=hf, verbose=False,
        l_trans_bandwidth=tb, h_trans_bandwidth=tb)


def k_under_interpolate(data, k):

    x = np.where(data > k)[0]
    y = data[x]
    f = interp1d(x, y, fill_value='extrapolate')
    data = f(np.arange(0, len(data), 1))

    return data 

def trim_signal(x, min_, max_):
    x[x> max_] = np.log(np.abs(x[x> max_]))+max_
    x[x< min_] = -np.log(np.abs(x[x< min_]))+min_
    return x

def standardization_signal(x):

    center = np.mean(x)
    scale = np.std(x)
    if scale == 0: scale = 1.0

    return (x - center) / scale


def remove_mv(x, kernel_size):

    center = fftconvolve(x, np.ones(shape=(kernel_size,))/kernel_size, mode='same')
    center[np.isnan(center) | np.isinf(center)] = 0.0 
    return x - center

def remove_rms(x, kernel_size):

    # Compute and remove the rms with FFT convolution of squared signal
    temp = fftconvolve(np.square(x), np.ones(shape=(kernel_size,))/kernel_size, mode='same')

    # Deal with negative values (mathematically, it should never be negative, but fft artifacts can cause this)
    temp[temp < 0] = 0.0

    # Deal with invalid values
    invalidIndices = np.isnan(temp) | np.isinf(temp)
    temp[invalidIndices] = 0.0
    maxTemp = np.max(temp)
    temp[invalidIndices] = maxTemp

    scale = np.sqrt(temp)
    # To correct for record 12 that has a zero amplitude chest signal
    scale[(scale == 0) | np.isinf(scale) | np.isnan(scale)] = 1.0

    return x / scale