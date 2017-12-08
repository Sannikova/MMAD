from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

from mfcc_features import mfcc as sg_mfcc
from functions import create_params,Derivative
import functions_for_mfcc as fm

(rate,sig) = wav.read("E:\\usr00404_male_adult_004.wav")

def mfcc_feat(signal,rate):
    return mfcc(signal,rate,winlen=0.025,winstep=0.01,
                 numcep=13,nfilt=13,nfft=1200,lowfreq=300,highfreq=8000,
                 preemph=0,ceplifter=0,appendEnergy=False)
    

def our_mfcc(x, sampleRate, blockDuration=0.025, blockOverlap=0.01,
    minFrequency=0, maxFrequency=None, mfccCount=13,
    signalWindow=None, spectrumWindow=None, squareSpectrum=None):

    maxFrequency = sampleRate//2
    
    fftSize = fm.sampleCount(blockDuration, sampleRate)
    stepSize = fftSize - fm.sampleCount(blockOverlap, sampleRate)
   
    filters = np.array(fm.generateFilters(minFrequency, maxFrequency,
        mfccCount, sampleRate, fftSize),dtype=float)

    
    _mfcc = np.array([])

    for i in range(0, len(x) - fftSize, stepSize):

        block = x[i:i+fftSize]
        
        if signalWindow:
            block *= signalWindow
        
        sp = (np.square(np.absolute(
            np.fft.rfft(block, n=fftSize))) / fftSize)[:fftSize//2]

        if spectrumWindow:
            sp *= spectrumWindow

        if squareSpectrum:
            sp = np.square(sp)
            
        S = [np.sum(sp * filters[j]) for j in range(mfccCount)]
        
        _mfcc = np.concatenate((_mfcc, np.array(fm.dctNorm(np.log(S)))))
        
    return _mfcc.reshape((-1, mfccCount))



def IS(ready_mfcc):
    mfcc1 = mfcc2 = ready_mfcc
    m = ready_mfcc.shape[1]
    
    info_params = np.zeros([ready_mfcc.shape[1],(ready_mfcc.shape[1]*6)-6])
    info_params[0,:m] = create_params(ready_mfcc,ready_mfcc.shape[1],m_mean = True)
    info_params[0,m:m*2] = create_params(ready_mfcc[:m],ready_mfcc.shape[1],m_dispersion = True) 
    
    mfcc_der1 = np.array(Derivative(mfcc1,fDer = True, sDer = False))
    info_params[0,m*2:m*3-1] = create_params(mfcc_der1,mfcc_der1.shape[1],m_mean = True)
    info_params[0,m*3-1:m*4-2] = create_params(mfcc_der1,mfcc_der1.shape[1],m_dispersion = True)
   
    mfcc_der2 = np.array(Derivative(mfcc2,fDer = False, sDer = True))
    info_params[0,m*4-2:m*5-4] = create_params(mfcc_der2,len(mfcc_der2),m_mean = True) 
    info_params[0,m*5-4:] = create_params(mfcc_der2,len(mfcc_der2),m_dispersion = True)
    
    return info_params[0]

def get_mfcc(x, sampleRate):
    return IS(our_mfcc(x,sampleRate))


