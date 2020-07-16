## @package Datagenerator
# contains the basic Data Generator

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview


import numpy as np
import os
from matplotlib import pyplot as plt
# for saving figures:
#import pickle

## 
# 
# generates windows with 8 basic signals, variation and Noise
class DataGenerator():
    'Generates Windows'
    def __init__(self,n_samples= 100,resolution = 0.01,SNR_dB = 1000, variation = 1, n_classes=8, useseed = False, seed = 5):
        'Initialization'
        self.n_samples = n_samples
        self.resolution = resolution
        self.windowsize = n_samples*resolution#(n_samples-1)
        self.SNR_dB = SNR_dB
        self.variation = variation
        self.n_classes = n_classes
        self.useseed = useseed
        self.seed = seed
        self.time = np.linspace(0, self.windowsize, self.n_samples) 
        class_names = ['const', 'rise', 'fall', 'sine', 'dirac up', 'dirac down', 'step up', 'step down']
        self.class_names = class_names[:n_classes]

    ## adds Noise to the Signal according to the given SNR
    def addNoise(self,signal, SNR_dB):
        ## Calculate signal power and convert to dB 
        x_watts = signal ** 2
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        ## Calculate noise according to signal then convert to watts
        noise_avg_db = sig_avg_db - SNR_dB
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        ## Generate an sample of white noise
        mean_noise = 0
        noise= np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
        ## return Noised up original signal
        return signal + noise

    ## return positive random number in a certain range \n
    # for variation = 1 between min and max \n
    # for variation = 0 mean value
    def randomnumber(self,min, max,middle,variation):
        return abs(self.randomsignednumber(min, max, middle, variation))

    ## returns random signed number in a certain range \n
    # for variation = 1 between min and max \n
    # for variation = 0 mean value \n
    # Sign is random

    def randomsignednumber(self,min, max, middle, variation):
        max_span = max-middle
        min_span = middle-min
        if self.useseed == True:
            np.random.seed(self.seed)
        if np.random.rand() < 0.5: # half max, half min
            if self.useseed == True:
                np.random.seed(self.seed)
            number = middle + max_span*np.random.rand()*variation
        else:
            if self.useseed == True:
                np.random.seed(self.seed)
            number = middle - min_span*np.random.rand()*variation
        if self.useseed == True:
            np.random.seed(self.seed)
        sign = 1 if np.random.rand() < 0.5 else -1
        return sign*number

    ## generates a window of the shape y_func
    def generate_random_sample(self, y_func):
        x = np.linspace(1/self.n_samples, 1, self.n_samples) 
        y = y_func(x)-y_func(0)  #+offset
        if self.SNR_dB == 1000:
            y_noised = y
        elif not np.any(y): #only zeros
            #noise = SNR_dB*np.random.rand(x.size)-(SNR_dB/2) 
            y_noised = self.addNoise(y+10,self.SNR_dB)-10 #TODO Anpassen bei Signal 0
        else:
            y_noised = self.addNoise(y,self.SNR_dB)
        return (y_noised)

    ## 0
    def generate_random_sample_const(self):
        y_func = lambda x: x-x
        return self.generate_random_sample(y_func)

    ## 1
    def generate_random_sample_rise(self):
        a = self.randomnumber(1, 100, 5, self.variation)
        y_func = lambda x: a*x
        return self.generate_random_sample(y_func) 

    ## 2
    def generate_random_sample_fall(self):
        y = self.generate_random_sample_rise()
        return -y 

    ## 3
    def generate_random_sample_trig(self):
        minamp = 0.5
        maxamp = 50
        a = self.randomsignednumber(minamp, maxamp, 1, self.variation) #Amplitude
        minfreq = 1#0.5
        #maxfreq = self.n_samples/2
        maxfreq = self.n_samples/5
        #maxfreq = 6
        b = self.randomsignednumber(minfreq, maxfreq, 1, self.variation) #frequency (Hz)
        c = self.randomsignednumber(-1/2, 1/2, 0, self.variation) #shift
        y_func = lambda x: a*np.sin(2*np.pi*b*(x-c))
        return self.generate_random_sample(y_func)

    ## 4
    def generate_random_sample_dirac_up(self): 
        min = 0.00002
        max = 0.001
        a = self.randomnumber(min, max, 0.0002, self.variation) 
        if a <= min:
            a = min
        buffer = 3/self.n_samples+np.sqrt(a)*2
        b = self.randomnumber(buffer, 1-buffer, 1/2, self.variation) 
        if b < buffer:
            b = buffer
        if b > 1-buffer:
            b = 1-buffer
        y_func = lambda x: (1/(np.sqrt(2*np.pi*a)))*np.exp(-(((x-b)**(2))/(2*a)))
        return self.generate_random_sample(y_func)

    ## 5
    def generate_random_sample_dirac_down(self):
        y = self.generate_random_sample_dirac_up()
        return -y

    ## 6
    def generate_random_sample_step_up(self):
        a = self.randomnumber(0.1, 10, 1, self.variation) #50
        buffer = 3/self.n_samples
        b = self.randomnumber(buffer, 1-buffer, 1/2, self.variation) 
        if b < buffer:
            b = buffer
        if b > 1-buffer:
            b = 1-buffer
        #y_func = lambda x: a if x>b else 0
        def y_func(x):
            y = a*np.heaviside((x-b),1)
            return y
        return self.generate_random_sample(y_func)

    ## 7
    def generate_random_sample_step_down(self):
        y = self.generate_random_sample_step_up()
        return -y

    def __str__(self):
        if self.SNR_dB == 1000:
            print("No Noise")
        else:
            print("Siganl to Noise Ratio = " +str(self.SNR_dB)+ "dB")
        print("Variation = " + str(self.variation*100)+ "%")
        print("Window-length = "+ str(self.windowsize))
        print("Number of labels = " + str(self.n_classes))

    ##function to generate a random window
    def __data_generation(self):
        if self.useseed == True:
            np.random.seed(self.seed)
        signal_form = int(np.floor(self.n_classes*np.random.rand())) #Alle Formen
        if signal_form == 0:
            window = self.generate_random_sample_const()
        elif signal_form == 1:
            window = self.generate_random_sample_rise()
        elif signal_form == 2:
            window = self.generate_random_sample_fall()
        elif signal_form == 3:
            window = self.generate_random_sample_trig()
        elif signal_form == 4:
            window = self.generate_random_sample_dirac_up()
        elif signal_form == 5:
            window = self.generate_random_sample_dirac_down()
        elif signal_form == 6:
            window = self.generate_random_sample_step_up()
        elif signal_form == 7:
            window = self.generate_random_sample_step_down()
        else :
            print("ERROR: Signal Form " + str(signal_form) +" not available")
        label = signal_form
        
        return window, label
    
    ## public function to get 1 window
    def getitem(self):
        'Generate one Window'

        # Generate data
        window, label = self.__data_generation()

        return window, label, self.time, self.class_names
    
    ## generator
    def genWindow(self,x):
        for i in range(x):
            window, label = self.__data_generation()
            yield window,label







    ########################
    ###testing functions###
    ########################
    
    def const_test(self,n_windows):
        X_test= np.empty([n_windows,self.n_samples])
        for i in range(0,n_windows):
            y_func = lambda x: x-x
            X_test[i] = self.generate_random_sample(y_func)
        return X_test

    def rise_test(self,n_windows):
        a = np.linspace(1, 100,n_windows)
        X_test= np.empty([n_windows,self.n_samples])
        for i in range(0,n_windows):
            y_func = lambda x: a[i]*x
            X_test[i] = self.generate_random_sample(y_func)
        return X_test

    def trig_test(self,n_windows):
        partintervall = int(n_windows**(1/3))
        a = np.linspace(0.5, 50,partintervall)
        b = np.linspace(0.5,self.n_samples/5,partintervall)
        c = np.linspace(-0.5,0.5,partintervall)
        X_test= np.empty([n_windows,self.n_samples])
        index = 0
        for i in a:
            for j in b:
                for k in c:
                    y_func = lambda x: i*np.sin(2*np.pi*j*(x-k))
                    X_test[index] = self.generate_random_sample(y_func)
                    index+=1

        for i in range(index,n_windows): #filling up
            y_func = lambda x: 1*np.sin(2*np.pi*1*(x-0))
            X_test[i] = self.generate_random_sample(y_func)
        return X_test

    def dirac_up_test(self,n_windows):
        partintervall = int(n_windows**(1/2))
        a_min = 0.00002
        a_max = 0.001
        a = np.linspace(a_min, a_max, partintervall) 
        X_test= np.empty([n_windows,self.n_samples])

        index = 0
        for i in a:
            buffer = 3/self.n_samples+np.sqrt(i)*2
            b = np.linspace(buffer, 1-buffer, partintervall) 
            for j in b:
                y_func = lambda x: (1/(np.sqrt(2*np.pi*i)))*np.exp(-(((x-j)**(2))/(2*i)))      
                X_test[index] = self.generate_random_sample(y_func)
                index+=1

        for i in range(index,n_windows): #filling up
            y_func = lambda x: (1/(np.sqrt(2*np.pi*0.0002)))*np.exp(-(((x-0.5)**(2))/(2*0.0002)))
            X_test[i] = self.generate_random_sample(y_func)
        return X_test

    def step_up_test(self,n_windows):
        partintervall = int(n_windows**(1/2))

        a = np.linspace(0.1,10,partintervall)
        buffer = 3/self.n_samples
        b = np.linspace(buffer,1-buffer,partintervall)
        X_test= np.empty([n_windows,self.n_samples])

        index = 0
        for i in a:
            for j in b:
                y_func = lambda x: i*np.heaviside((x-j),1)  
                X_test[index] = self.generate_random_sample(y_func)
                index+=1

        for i in range(index,n_windows): #filling up
            y_func = lambda x: 1*np.heaviside((x-0.5),1)
            X_test[i] = self.generate_random_sample(y_func)
        return X_test

    def generatetestsignal(self,n_windows,datafile = ''):         
        X_test = np.empty([n_windows,self.n_samples])
        Y_test = np.empty(n_windows)
        intervall = int(n_windows/8)

        #0 const
        X_test[:intervall,:] = self.const_test(intervall) 
        Y_test[:intervall] = np.ones(intervall)*0
        
        #1 rise
        X_test[intervall:2*intervall,:] = self.rise_test(intervall) 
        Y_test[intervall:2*intervall] = np.ones(intervall)*1
        
        #2 fall
        X_test[2*intervall:3*intervall,:] = -self.rise_test(intervall) 
        Y_test[2*intervall:3*intervall] = np.ones(intervall)*2

        #3 trig
        X_test[3*intervall:4*intervall,:] = self.trig_test(intervall) 
        Y_test[3*intervall:4*intervall] = np.ones(intervall)*3

        #4 dirac up
        X_test[4*intervall:5*intervall,:] = self.dirac_up_test(intervall) 
        Y_test[4*intervall:5*intervall] = np.ones(intervall)*4

        #5 dirac down
        X_test[5*intervall:6*intervall,:] = -self.dirac_up_test(intervall) 
        Y_test[5*intervall:6*intervall] = np.ones(intervall)*5

        #6 step up
        X_test[6*intervall:7*intervall,:] = self.step_up_test(intervall) 
        Y_test[6*intervall:7*intervall] = np.ones(intervall)*6

        #7 step down
        X_test[7*intervall:8*intervall,:] = -self.step_up_test(intervall) 
        Y_test[7*intervall:8*intervall] = np.ones(intervall)*7

        for i in range(8*intervall,n_windows):
            X_test[i,:] = self.const_test(1)
            Y_test[i] = 0

        #Data = np.empty([2,n_windows,self.n_samples])
        #Data[0,:,:] = X_test
        #Data[1,:,:] = np.ones(self.n_samples)* Y_test
        #Data[1,:,:] =Y_test
        #if datafile != '':
        #    np.save(datafile,Data)
        return X_test,Y_test

