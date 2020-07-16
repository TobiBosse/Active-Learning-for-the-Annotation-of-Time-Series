## @package Datagenerator2dontcare
# contains the imbalanced Datagenerator

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview


import numpy as np
import os
from matplotlib import pyplot as plt
from Datagenerator import DataGenerator 
# for rectangular signal
from scipy import signal
# for saving figures:
#import pickle

# Splitting of generated Dataset
from sklearn.model_selection import train_test_split

## This Generator extends the basic Generator with more shapes and produces imbalanced Datasets \n
#   advanced shapes are documented in Thesis \n
#   testing Functions are not complete yet: The idea behnid this was to fill the space in Feature Space for each class. Which means it produces every window possible with a variable step width \n
#   test functions for the basic signals are in the basic Generator
class DataGenerator2dontcare(DataGenerator):
    'Generates Windows'
    def __init__(self,n_samples= 100,resolution = 0.01,SNR_dB = 1000, variation = 1, n_channels=1, n_classes=5, shuffle=False, useseed = False, seed = 5):
        'Initialization'
        super().__init__(n_samples, resolution, SNR_dB, variation, n_classes, useseed, seed)
        self.n_channels = n_channels
        #base_classes = ['const', 'rise', 'fall', 'sine', 'dirac up', 'dirac down', 'step up', 'step down', 'rect', 'triang', 'tan', 'e', 'poly', 'parabel']
        #base_classes = ['const', 'rise', 'fall', 'sine', 'dirac up', 'dirac down', 'step up', 'step down', 'rect', 'triang', 'tan', 'e', 'poly', 'parabel','spine0' , 'spine1', 'spine2']
        self.class_names = ['pos const', 'neg const', 'low rise', 'neg fall', 'low sine', 'dont care']
        #self.class_names = base_classes*3
        #for i in range(0,int(n_classes/3)):
        #    self.class_names[i*3] = 'pos ' + base_classes[i] 
        #    self.class_names[i*3+1] = 'low ' + base_classes[i] 
        #    self.class_names[i*3+2] = 'neg ' + base_classes[i] 

    ## test signals
    
    ## 8
    def generate_rectangular_pulse(self):
        minamp = 0.5
        maxamp = 50
        a = self.randomsignednumber(minamp, maxamp, 1, self.variation) #Amplitude
        minfreq = 1
        maxfreq = self.n_samples/10
        b = self.randomsignednumber(minfreq, maxfreq, 1, self.variation) #frequency (Hz)
        c = self.randomsignednumber(-1/2, 1/2, 0, self.variation) #shift
        y_func = lambda x: a*signal.square(2 * np.pi * b * (x-c))
        return self.generate_random_sample(y_func)
    
    # 9
    def generate_triangular_function(self):
        x = np.linspace(1/self.n_samples, 1, self.n_samples) 
        a = self.randomnumber(1, 100, 5, self.variation)
        parts = int(np.floor(4*np.random.rand()))+2

        y = a *x
        for p in range(1,parts):
            y[int((p/parts)*self.n_samples):] = a*x[int((p/parts)*self.n_samples):] - a*x[int((p/parts)*self.n_samples)-1]

        if self.SNR_dB == 1000:
            y_noised = y
        else:
            y_noised = self.addNoise(y,self.SNR_dB)
        return (y_noised)
    
    ## 10
    def generate_tan(self):
        y_func = lambda x: np.tan(np.pi*x)
        return self.generate_random_sample(y_func)

    ## 11
    def generate_e(self):
        y_func = lambda x: np.exp(2*x)
        return self.generate_random_sample(y_func)

    ## 12
    def generate_poly(self):
        y_func = lambda x: x**4 + 2*x**3 - 4*x**2 +2*x
        return self.generate_random_sample(y_func)

    ## 13
    def generate_parabel(self):
        y_func = lambda x: (x)**2
        return self.generate_random_sample(y_func)

    ## 14
    def generate_spine0(self):
        y_func = lambda x: (8*(x-0.5)-5)**2+(8*(x-0.5)-5)*(8*(x-0.5)**2-5)+(8*(x-0.5)**2-5)**2
        return self.generate_random_sample(y_func)

    ## 15
    def generate_spine1(self):
        y = self.generate_spine0()/80
        return y

    ## 16
    def generate_spine2(self):
        y = -self.generate_spine0()/200 +1
        return y

    ## Function that is used to generate a window
    def __data_generation(self):
        if self.useseed == True:
            np.random.seed(self.seed)
        signal_form = int(np.floor(12)*np.random.rand()) #Alle Formen
        height = int(np.floor(3*np.random.rand())) # pos, low, neg

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
        elif signal_form == 8:
            window = self.generate_rectangular_pulse()
        elif signal_form == 9:
            window = self.generate_triangular_function()
        elif signal_form == 10:
            window = self.generate_tan() 
        elif signal_form == 11:
            window = self.generate_e()
        elif signal_form == 12:        
            window = self.generate_poly()
        elif signal_form == 13: 
            window = self.generate_parabel()
        elif signal_form == 14: 
            window = self.generate_spine0()
        elif signal_form == 15: 
            window = self.generate_spine1()
        elif signal_form == 16: 
            window = self.generate_spine2()
        else :
            print("ERROR: Signal Form " + str(signal_form) +" not available")
        
        if height == 0:
            start = self.randomnumber(100,500,250,1)    
        elif height == 1:
            start = self.randomsignednumber(-100,100,0,1)
        elif height == 2:
            start = -self.randomnumber(100,500,250,1)
        else:
            print("ERROR: Signal Height " + str(height) +" not available")

        window = start + window ## add offset

        if height == 0 and signal_form == 0:
            label = 0 #'pos const'
        elif height == 2 and signal_form ==0:
            label = 1 #'neg const'
        elif height == 1 and signal_form == 1:
            label = 2 #'low rise'
        elif height == 2 and signal_form == 2:
            label = 3 #'neg fall'
        elif height == 1 and signal_form == 3:
            label = 4 #'low sine'
        else:
            label = 5 # dont care
        
        label = np.clip(label,0,self.n_classes-1)

        return window, label
    
    ## generator \n 
    # yields x windows
    def genWindow(self,x):
        for i in range(x):
            window, label = self.__data_generation()
            yield window,label

    ## returns 1 long curve of stacked windows (behind each other)
    def gencurve(self,n_windows):
        #TODO numpy array instead of list
        signal = []
        label = np.empty(n_windows)
        signal.append(self.generate_random_sample_const()) # signal always starts with zeros
        label[0] = 0
        for i in range(1,n_windows):
            newSignal,label[i] = self.__data_generation()
            newSignal += signal[-1][-1] #add last Value to create continous signal
            signal.append(newSignal)

        #Conversion to numpy arraay
        longfunction = np.empty(0)
        for i in range(len(signal)):
            values = np.asarray(signal[i])
            longfunction= np.concatenate([longfunction,values])
        time = np.arange(0,longfunction.shape[0]*self.resolution, self.resolution)
        time_series = np.empty([2,longfunction.shape[0]])
        time_series[0] = time
        time_series[1] = longfunction
        return time_series,label
    
    ## returns a train and test set
    def GeneratePool(self, n_windows, useNeuralNet = False):
        """
        generates Pool of size <n_windows> and additional training data of size  <n_windows*test_size>
        and saves windows
        """
        test_size = 0.2 #20 % of training Data

        n_test_windows = int(n_windows*(test_size/(1-test_size)))
        n_total_windows = n_windows + n_test_windows
        #X = np.empty([n_total_windows,10])
        Y = np.empty(n_total_windows)
        windows = np.empty([n_total_windows,self.n_samples])
        i= 0
        for window,label in self.genWindow(n_total_windows): #n windows generieren 
            windows[i,:] = window
            Y[i] = label
            i+=1
            print(str(np.round((i/(n_total_windows))*100,3)) + " %" + " generiert", end = ' ')
            print("\r", end='')

        # Split
        train_windows, test_windows, Y_pool, Y_test  = train_test_split(windows,Y,test_size=test_size)

        return train_windows, test_windows, Y_pool, Y_test

    ## Testing functions

    ## fills the space of 
    def rect_test(self,n_windows):
        partintervall = int(n_windows**(1/3))
        a = np.linspace(0.5, 50,partintervall)
        b = np.linspace(1,self.n_samples/5,partintervall)
        c = np.linspace(-0.5,0.5,partintervall)
        X_test= np.empty([n_windows,self.n_samples])

        index = 0
        for i in a:
            for j in b:
                for k in c:
                    y_func = lambda x: i*signal.square(2 * np.pi * j * (x-k))
                    X_test[index] = self.generate_random_sample(y_func)
                    index+=1

        for i in range(index,n_windows): #filling up
            y_func = lambda x: 1*signal.square(2 * np.pi * 1 * (x-0))
            X_test[index] = self.generate_random_sample(y_func)
            index+=1
            
        return X_test

    ## Function Corpus to generate big testsignal, containing every shape and save it
    def generatetestsignal(self,n_windows,datafile = ''):  
        X_test= np.empty([n_windows,self.n_samples])
        Y_test = np.empty(n_windows)
        n_signals = 9
        intervall = int(n_windows/n_signals)

        #basic signals
        X_test[:8*intervall,:], Y_test[:8*intervall] = super().generatetestsignal(int(8*intervall))
        
        #8 rect
        X_test[8*intervall:9*intervall,:] = self.rect_test(intervall) 
        Y_test[8*intervall:9*intervall] = np.ones(intervall)*8

        for i in range(n_signals*intervall,n_windows):
            X_test[i,:] = super().const_test(1)
            Y_test[i] = 0

        #Data = np.empty([2,n_windows,self.n_samples])
        #Data[0,:,:] = X_test
        #Data[1,:,:] = np.ones(self.n_samples)* Y_test
        #Data[1,:,:] =Y_test
        #if datafile != '':
        #    np.save(datafile,Data)
        return X_test,Y_test


## Data generator #TODO
# Parameters
#params = {'dim': (32,32,32), 
#          'batch_size': 64,
#          'n_classes': 6,
#          'n_channels': 1,
#          'shuffle': True}

# Datasets
#partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
#labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

# Generators
#training_generator = DataGenerator(partition['train'], labels, **params)
#validation_generator = DataGenerator(partition['validation'], labels, **params)