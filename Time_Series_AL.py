## @package Time_Series_AL
# top level application: MAIN is here
##

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview

### basics
import numpy as np
import matplotlib.pyplot as plt

### AL Tool Collection and Data Generator
from ActiveLearningEnv.ActiveLearningTools import ActiveLearner

### Data Generator
from DataGenerators.Datagenerator2dontcare import ImbalancedDataGenerator

## Dummy class to wrap data loading functions 
class Dataloader():

    ## Converts Pandas Dataframes to Numpy
    def convertingtonumpy(self,x_pool, length = 100):

        size = x_pool.shape[0]
        numpyFormat = np.empty([size,length])    
        
        for i in range(size):
            for j in range(length):
                numpyFormat[i,j] = x_pool['dim_0'][i][j]


        return numpyFormat

    ## Converts UCR Datasets from http://www.timeseriesclassification.com/ to a useful format: \n
    # windows = Numpy matrix: Every row is one window \n
    # Labels = numpy array (1D)
    def postProcessSetfromTSCcom(self,windows_pool,windows_test, y_pool, y_test,window_length):
        #http://www.timeseriesclassification.com/
        #####################################################################################
        # handling of stupid Panda format

        windows_pool = self.convertingtonumpy(windows_pool,length = window_length)
        windows_test = self.convertingtonumpy(windows_test,length = window_length)

        y_pool_normal = np.empty(y_pool.reshape(-1).shape)
        for i,y in enumerate(y_pool):
            y_pool_normal[i] = y  
        y_pool = y_pool_normal 

        y_test_normal = np.empty(y_test.reshape(-1).shape)
        for i,y in enumerate(y_test):
            y_test_normal[i] = y
        y_test = y_test_normal
        #####################################################################################

        if y_pool.min() > 0 or y_test.min() > 0 :
            print("1st class is decoded as zero (was 1)")
            y_pool -= 1  #class 1 = class 0
            y_test -= 1

        return windows_pool, windows_test, y_pool, y_test

    ## Function to load a Dataset (Generated or UCR Dataset) and make it imbalance
    def getData(self,set_type): 
        ###################
        ## Data Generation
        ###################
        if set_type == "generator": 
            #Settings
            windowLength = 1
            samplesPerWindow = 100
            n_classes = 6
            n_windows = 1000
            Gen = ImbalancedDataGenerator(n_samples= samplesPerWindow,resolution = windowLength/samplesPerWindow,SNR_dB = 50, variation = 1, n_classes=n_classes, useseed = False, seed = 5)
            windows_pool, windows_test, y_pool, y_test = Gen.GeneratePool(n_windows)
            class_names = Gen.class_names

            ## Generator
            ##Data shape
            #n_samples = 100 #samples of 1 window (1 window exists of X samples)
            #resolution = 0.01 # time step between 2 samples
            #SNR_dB = 50 # Signal to Noise ration in dB
            #variation = 1 # 0 to 1 (0 to 100%), higher values possible
            #Gen = Datagenerator2.DataGenerator2(n_samples= n_samples,resolution = resolution,SNR_dB = SNR_dB, variation = variation, n_classes=30, useseed = False, seed = 5)
            
            # size var , length var, n classes 
            #if obj.fast_mode and not obj.singleErrorOutput:       
                #x_pool, x_test, y_pool, y_test = obj.GeneratePool(obj.n_windows)
                #train_windows = None
                #test_windows = None
            #else:
            #    windows_pool, windows_test, y_pool, y_test = Gen.GeneratePool(obj.n_windows)
            #    pass

        elif set_type == "GunPoint":
            ## Gunpoint Dataset
            windowLength = 1 #unspecified !
            samplesPerWindow = 50
            n_classes = 2
            n_windows = 50   
            from pyts.datasets import load_gunpoint
            windows_pool, windows_test, y_pool, y_test = load_gunpoint(return_X_y=True)  
            class_names = ['gun', 'point']
            if y_pool.min() > 0 or y_test.min() > 0 :
                print("1st class is decoded as zero (was 1)")
                y_pool -= 1  #class 1 = class 0
                y_test -= 1
            
        elif set_type == "Crop":
            #http://www.timeseriesclassification.com/description.php?Dataset=Crop
            windowLength = 1 #unspecified !
            samplesPerWindow = 46
            n_classes = 24
            n_windows = 7200 
            from sktime.utils.load_data import load_from_tsfile_to_dataframe
            windows_pool, y_pool = load_from_tsfile_to_dataframe("/home/tob/Datasets/Crop_TRAIN.ts")
            windows_test, y_test = load_from_tsfile_to_dataframe("/home/tob/Datasets/Crop_TEST.ts")
            windows_pool, windows_test, y_pool, y_test = self.postProcessSetfromTSCcom(windows_pool, windows_test, y_pool, y_test,samplesPerWindow)
            class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10', 'Class 11', 'Class 12', 'Class 13', 'Class 14', 'Class 15', 'Class 16', 'Class 17', 'Class 18', 'Class 19', 'Class 20', 'Class 21', 'Class 22', 'Class 23', 'Class 24']
            
        elif set_type == "FaceAll": 
            #http://www.timeseriesclassification.com/description.php?Dataset=FaceAll
            windowLength = 1 #unspecified !
            samplesPerWindow = 131
            n_classes = 14
            n_windows = 560
            from sktime.utils.load_data import load_from_tsfile_to_dataframe
            windows_pool, y_pool = load_from_tsfile_to_dataframe("/home/tob/Datasets/FaceAll_TRAIN.ts")
            windows_test, y_test = load_from_tsfile_to_dataframe("/home/tob/Datasets/FaceAll_TEST.ts")
            windows_pool, windows_test, y_pool, y_test = postProcessSetfromTSCcom(windows_pool, windows_test, y_pool, y_test,samplesPerWindow)
            class_names = ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 5', 'Student 6', 'Student 7', 'Student 8', 'Student 9', 'Student 10', 'Student 11', 'Student 12', 'Student 13', 'Student 14']

        elif set_type == "InsectWingbeat":
            #http://www.timeseriesclassification.com/description.php?Dataset=InsectWingbeat
            windowLength = 1 #unspecified !
            samplesPerWindow = 30
            n_classes = 10
            n_windows = 30000
            from sktime.utils.load_data import load_from_tsfile_to_dataframe
            windows_pool, y_pool = load_from_tsfile_to_dataframe("/home/tob/Datasets/InsectWingbeat_TRAIN.ts")
            windows_test, y_test = load_from_tsfile_to_dataframe("/home/tob/Datasets/InsectWingbeat_TEST.ts")
            windows_pool, windows_test, y_pool, y_test = postProcessSetfromTSCcom(windows_pool, windows_test, y_pool, y_test,samplesPerWindow)
            class_names = ['Insect 1', 'Insect 2', 'Insect 3', 'Insect 4', 'Insect 5', 'Insect 6', 'Insect 7', 'Insect 8', 'Insect 9', 'Insect 10']

        #make the Dataset imbalanced (class 0 to class 4 matters, class 5 don't care: exists of all other classes)
        #Setting
        n_classes = 6

        y_pool, y_test = np.clip(y_pool,0,n_classes-1), np.clip(y_test,0,n_classes-1)
        class_names = class_names[:n_classes]
        class_names[-1] = "don't care"
        print("imbalanced classes:")
        print(class_names)


        #reshaping for NN
        #if useNeuralNet == True:
        #    X = np.reshape(X, (X.shape[0],1,X.shape[1]))
        #    Y = np.reshape(Y, (Y.shape[0], 1))
        #    Y = to_categorical(Y,num_classes=self.n_classes)

        ## USE CASE 1 POOL : test = complete pool
        #x_test = np.copy(x_pool)
        #y_test = np.copy(y_pool)
        #test_windows = np.copy(train_windows)
        
        #obj.visualizeBoss(x_pool,y_pool)

        #Testing
        #print("Gen+ FST " + str(time.time() -test ))

        #End of Data Generation
        return windows_pool, windows_test, y_pool, y_test, class_names, windowLength, n_classes, samplesPerWindow

    ## Slim Function for real application
    def getPool(self,Type = "generator"):
        windows_pool, windows_test, y_pool, y_test, class_names, windowLength, n_classes, samplesPerWindow = self.getData(Type)
        return windows_pool, y_pool, class_names, windowLength, n_classes 

## Dummy User \n
# owns the knowledge
class Dummy_User():
    ## owns the knowledge and simulates User
    def __init__(self,y_pool,class_names):
        self.y_pool = y_pool # this is the knowledge /labels
        self.class_names = class_names

    def giveLabel(self,Window_ID):
        for label in self.y_pool[Window_ID]:
            print(self.class_names[int(label)],end = ' | ')
        print("")
        return self.y_pool[Window_ID]

    def givefirstLabels(self,number):
        initial_idx = np.random.choice(range(len(self.y_pool)), size= number, replace=False)  
        window_ID = initial_idx
        label = self.giveLabel(window_ID)
        return window_ID , label

## GUI
class UserInterface():
    def __init__(self,class_names,windowLength,user = None):
        self.class_names = class_names
        self.windowLength = windowLength
        self.user = user
        fig, axs = plt.subplots(10,1, figsize=(10, 50))
        #plt.title("windows queried") 
        #fig.subplots_adjust(hspace = .5, wspace=.001)
        self.axs = axs.ravel()

    def showWindows(self,windows,labels):
        with plt.style.context('seaborn-white'):
            for i in range(10): #10 is choosen bagsize
                window = windows[i]
                valuesPerWindow = window.shape[0]
                self.axs[i].cla()
                self.axs[i].scatter(np.linspace(0,self.windowLength,valuesPerWindow),window)
                self.axs[i].plot(np.linspace(0,self.windowLength,valuesPerWindow),window,linewidth = 0.5,linestyle = 'dashed')           
            plt.draw()
            plt.pause(0.01)
        for label in labels:
            print(self.class_names[int(label)],end = ' | ')      
        print("")
        
    def getlabel(self,windows,labels,windowIDs = None):  # Window ID is only here because dummy user needs it
        self.showWindows(windows,labels)
        return self.user.giveLabel(windowIDs)

## MAIN: Real run, structured as explained in the thesis
def LabelingProcess(windows_pool, class_names, windowLength, n_classes, user,algo = "ROCKET", dontcareLimit = 50, labels = None):
    print("inititalize")
    if labels is None: 
        coldStart = True
    else:
        coldStart = False
        first_window_IDs = np.where(labels != -1)
        first_labels = labels[first_window_IDs]

    #create GUI
    GUI = UserInterface(class_names,windowLength,user)
    class_names,windowLength,user = None,None,None # user can only be reached through the GUI

    # create Storage
    labels = np.ones(windows_pool.shape[0])*-1
        # -1 is no label
        # 0 is first class, last is don't care
        # Names in list "class_names"

    print("Feature Space Transformation")
    # Create Learner Object
    Learner = ActiveLearner(windows_pool,n_classes,algo) 

    ######################################
    ## Begin
    ######################################

    #receive fist few labels
    if coldStart == True:
        first_window_IDs, first_labels  = GUI.user.givefirstLabels(50)

    #store labels
    labels[first_window_IDs] = first_labels

    
    ## workaround for problem with less samples: At least 2 different classes must exist, e.g. one interesting class and don't care
    # can be removed for BOSS start
    while first_labels.max() == first_labels.min(): #all the same class    
        print("all windows are the same class, try again")
        first_window_IDs, first_labels  = user.givefirstLabels(50)

    #inital training
    Learner.initialTraining(first_window_IDs,first_labels)

    #first Query
    windowIDs, predictedLabels = Learner.query()

    dontCareCounter = 0
    exit = False

    ## Active Learning Process
    while not exit:
        # get label for windows
        windows = windows_pool[windowIDs]
        realLabels = GUI.getlabel(windows,predictedLabels,windowIDs)
        
        #store new labels
        labels[windowIDs] = realLabels

        ## Output 
        print("predicted (top) and Real (bottom)")
        print(predictedLabels.astype(np.int64))
        
        print(realLabels.astype(np.int64))
        print("")
    
        ## check Exit Criteria
            # 50 samples of dont care in a row
        if realLabels.min() == n_classes -1: # all dont care
            dontCareCounter += realLabels.shape[0] #choosen bagsize
        else: 
            dontCareCounter = 0
        if dontCareCounter >= dontcareLimit or labels.min() > -1: # All samples labeled, no sample has label -1
            exit = True

        # query new windows
        if not exit:
            windowIDs, predictedLabels = Learner.ActiveLearningIteration(realLabels)
        else :
            print("All interesting windows found")
            break
 
    return windows_pool,labels
        

###################################
#    -           __
#  --          ~( @\   \
# ---   _________]_[__/_>________
#      /  ____ \ <>     |  ____  \
#     =\_/ __ \_\_______|_/ __ \__D
# ________(__)_____________(__)____
###################################
## Example of Application
###################################

## load Data
print("load Data")
## pool contains all windows, y_pool all labels, class_names the name of the classes, windowLength the duration of 1 window in seconds, n_classes the number of different classes existing.
windows_pool, y_pool, class_names, windowLength, n_classes = Dataloader().getPool("generator") #Crop , FaceAll, GunPoint

## create dummy user
user = Dummy_User(y_pool,class_names) # User has the knowledge
y_pool = None # delete knowledge


## only ROCKET
#windows, labels = LabelingProcess(windows_pool, class_names, windowLength, n_classes, user)

## Combination
# start with BOSS
windows, labels = LabelingProcess(windows_pool, class_names, windowLength, n_classes, user, algo = "BOSS", dontcareLimit= 10)

# switch to ROCKET
windows, labels = LabelingProcess(windows_pool, class_names, windowLength, n_classes, user, labels = labels) #, algo = "ROCKET", dontcareLimit= 50

## Test success
labels[np.where(labels == -1)] = n_classes -1  # label windows without label with "dont care"
difference = np.where(labels != user.y_pool)
print(labels[difference])
print(user.y_pool[difference])
