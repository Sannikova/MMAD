from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

from functions import cepweight,firstDerivative,secondDerivative,create_params,Derivative
import functions_for_mfcc as fm

(rate,sig) = wav.read("E:\\usr0001_male_youth_008.wav")

#Вычисление мел-кепстральных коэффициентов с помощью библиотеки python_speech_features
#mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,
#                 numcep=13,nfilt=13,nfft=1200,lowfreq=300,highfreq=8000,
#                 preemph=0,ceplifter=0,appendEnergy=False)
#print('MFCC: ',mfcc_feat)

def our_mfcc(x, sampleRate, blockDuration=0.025, blockOverlap=0.01,
    minFrequency=300, maxFrequency=8000, mfccCount=13,
    signalWindow=None, spectrumWindow=None, squareSpectrum=None):

    fftSize = fm.sampleCount(blockDuration, sampleRate)
    stepSize =fftSize- fm.sampleCount(blockOverlap, sampleRate)

    filters = fm.generateFilters(minFrequency, maxFrequency,
        mfccCount, sampleRate, fftSize)

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
            
        S = [np.log(np.sum(sp * filters[j])) for j in range(mfccCount)]

        _mfcc = np.concatenate((_mfcc, np.array(fm.dctNorm(S))))
        
    return _mfcc.reshape((-1, mfccCount))

our_mfcc_feat = our_mfcc(sig,rate)


def IS(_mfcc):
    
    mfcc1 = _mfcc; mfcc2 = _mfcc
    info_params = np.zeros([_mfcc.shape[1],_mfcc.shape[1]*6])
    print('orig mfcc',_mfcc)
    info_params[0,:13] = create_params(_mfcc,_mfcc.shape[1],m_mean = True)
    info_params[0,13:26] = create_params(_mfcc,_mfcc.shape[1],m_dispersion = True)
    
    mfcc_der1 = np.array(([firstDerivative(cepweight(mfcc1[i])) for i in range(mfcc1.shape[1]-1)]))
    print('\nNOorig mfcc',_mfcc)
   # mfcc_der1 = Derivative(mfcc1,fDer = True, sDer = False)   
    info_params[0,26:38] = create_params(mfcc_der1,len(mfcc_der1),m_mean = True)
    info_params[0,38:50] = create_params(mfcc_der1,len(mfcc_der1),m_dispersion = True)
    #print('\n\nmfcc',_mfcc,'\n\nmfcc2',mfcc2)
    print('\nNOorig mfcc',_mfcc)
    mfcc_der2 = np.array([secondDerivative(cepweight(mfcc2[i:])) for i in range(mfcc2.shape[1])])
    
    info_params[0,50:65] = create_params(mfcc_der2,len(mfcc_der2),m_mean = True) 
    info_params[0,65:] = create_params(mfcc_der2,len(mfcc_der2),m_dispersion = True)
    
    return info_params[0]


characteristic = IS(our_mfcc_feat)
print(characteristic)


