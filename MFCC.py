from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.fft import fft, rfftfreq
import math


(rate,sig) = wav.read("D:\\usr0001_male_youth_008.wav")
#print('rate: ',rate)
#print('signal: ',sig)
#print('len signal: ', len(sig))

#Вычисление мел-кепстральных коэффициентов с помощью библиотеки python_speech_features
mfcc_feat = mfcc(sig,rate)
print('MFCC: ',len(mfcc_feat))

#Пишем MFCC сами

#1.Разложение в ряд Фурье
N=256#Длина фрейма
X = fft(sig,N)
for k in range(N-1):
    Hemming_window = 0.54 - 0.46*np.cos(2*np.pi*k/(N-1))
    X[k]=X[k]*Hemming_window
     
#2.Расчет мел-фильтров
def toMel(F):#Функция преобразования частоты в мел
    M=1127*np.log(1+F/700)
    return M

def toFrequency(M):#Функция преобразования мел в частоту
    F = 700*(pow(np.e,(M/1127))-1)
    return F

M = 13 #Кол-во мел-опорных коэффициентов
F_n = M + 2 #Количество опортных точек
m = np.zeros((F_n))
F_start = toMel(300) #Начальное значение мел-интервала
F_finish = toMel(8000) #Конечное значение
F_step = (F_finish - F_start)/(F_n-1) #Шаг между значениями
F=F_start

for i in range(F_n):
    m[i]=F
    F=F+F_step
#print('m[i]: ', m)

h = np.zeros((F_n))
for i in range(F_n):
    h[i] = toFrequency(m[i])
#print('h[i]: ', h)

f = np.zeros((F_n))
sampleRate = rate
frameSize = N
for i in range(F_n):
    temp = (frameSize+1)*h[i]/sampleRate
    f[i]=math.floor(temp)
#print('f[i]: ', f)

H = np.ones(((M,N)))
for m in range(M):
    for k in range(N):
        #print('m:',m,'f[m]:',f[m], 'f[m-1]:', f[m-1],'f[m+1]:',f[m+1],'k:',k)
        if ((m>0)&(k<f[m-1])): 
            H[m][k]=0
            #print('Hm[k] прошло 1 фильтр')
        if (f[m-1] <= k <=f[m]) : 
            H[m][k] = (k-f[m-1])/(f[m]-f[m-1])
            #print('Hm[k] прошло 2 фильтр',H[m][k])
        if (f[m]< k <=f[m+1]):
            H[m][k] = ((f[m+1]-k)/(f[m+1]-f[m]))
            #print('Hm[k] прошло 3 фильтр',H[m][k])
        if ((m<M-1)&(k > f[m+1])): 
            H[m][k]=0
            #print('Hm[k] прошло 4 фильтр')
#print('Hm(k): ',H)            

#3.Логарифмирование энергии спектра
S=np.ones((M))
for m in range(M):
    temp_sum=0
    for k in range(N):
        temp_sum+=pow(abs(X[k]),2)*H[m][k]
    S[m]=np.log(temp_sum)
    #print('m:', m, 'k:',k,'S[m]:',S[m])
#print('S[m]:',S)

##4.Косинусное преобразование
C = np.ones((M))
for l in range(M):
    temp_sum = 0
    for m in range(M):
        temp_sum+=S[m]*np.cos((np.pi*l*(m+0.5))/M)
    C[l]=temp_sum
print('C: ',C)


















  