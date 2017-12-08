import numpy as np

def mel(f):
    #TODO:
    return 1127*np.log(1+f/700)

def hz(m):
    #TODO: just do it
    return 700*(pow(np.e,(m/1127))-1)

def bin(f,fftSize,sampleRate):
    #TODO: just do it
    return int(f*fftSize/sampleRate)

def generateFilters(minF,maxF,count,sampleRate,fftSize):
    #TODO: just do it
    melpoints = np.linspace(mel(minF),mel(maxF),count+2) 
    points=[bin(hz(m),fftSize,sampleRate) for m in melpoints]
    
    filters = np.zeros((count, fftSize // 2))

    for i in range(1, count + 1):
            for k in range(filters.shape[1]):
                value = 0
    
                if points[i - 1] <= k <= points[i]:
                    value = (k - points[i - 1]) / (points[i] - points[i - 1])
                elif points[i] <= k <= points[i + 1]:
                    value = (points[i + 1] - k) / (points[i + 1] - points[i])
    
                filters[i-1, k] = value
    return filters    

# дискретное косинусное преобразование-2 с нормировкой
def dctNorm(x):
    N = len(x)
    y = np.zeros(N)

    for k in range(N):
        items = [x[n] * np.cos(np.pi * k * (n + 0.5) / N) for n in range(N)]
        value = 2 * np.sum(items)
        value *= np.sqrt(1 / (4 * N)) if k == 0 else np.sqrt(1 / (2 * N))

        y[k] = value

    return y

def sampleCount(duration, sampleRate):
    return int(duration * sampleRate)

def signalWindowSize(duration, sampleRate):
    return sampleCount(duration, sampleRate)

def spectrumWindowSize(duration, sampleRate):
    return signalWindowSize(duration, sampleRate) // 2

def hemmingWindow(M):
    return 0.54-0.46*np.cos((2*np.pi*np.arange(0,M)/(M-1)))