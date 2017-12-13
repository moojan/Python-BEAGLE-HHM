
import numpy as np

def cconv(first, second):
    return np.real(np.fft.ifft(np.fft.fft (first) * np.fft.fft (second)))



