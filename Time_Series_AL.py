## @package Time_Series_AL
# top level application: MAIN is here
##

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview


import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn import metrics

### AL Tool Collection
import ActiveLearningTools as ALtools

### Data Generator
#import Datagenerator2
import Datagenerator2dontcare


## calculate Accuracy from Confusion Matrix
def calculate_accuracy(Confusionmatrix,sample_weight= None):
    Durchlaeufe = Confusionmatrix.shape[0]
    questions = Confusionmatrix.shape[-1]
    n_classes = Confusionmatrix.shape[1]
    if n_classes != Confusionmatrix.shape[2]:
        print("ERROR: Confusion matrix not sqare-matrix")

    accuracy_scores = np.empty([Durchlaeufe,questions])
    for i in range(Durchlaeufe):  ##i = Test_run
        for j in range(questions): #j = question
            y_true = np.empty(0)
            y_pred = np.empty(0)
            for k in range(n_classes): # Y True
                for l in range(n_classes): #Y_pred
                    for m in range(int(Confusionmatrix[i,k,l,j])):
                        y_true = np.append(y_true,k)
                        y_pred = np.append(y_pred,l)
            accuracy_scores[i,j] = metrics.accuracy_score(y_true,y_pred,sample_weight=sample_weight)
        
    #accuracy_scores = accuracy_scores/Durchlaeufe
    return accuracy_scores

## Converts Numpy arrays to Pandas Dataframes
#   #def convertingtopanda(X,Y=None,Z=None):
    #    print(X.shape)
    #    X = pd.DataFrame(pd.DataFrame(X).apply(lambda x: [np.array(x)], axis=1).apply(lambda x: x[0]))
    #    if Y is None:
    #        return X
    #    Y = pd.DataFrame(pd.DataFrame(Y).apply(lambda x: [np.array(x)], axis=1).apply(lambda x: x[0]))
    #    if Z is None:
    #        return X,Y
    #    Z = pd.DataFrame(pd.DataFrame(Z).apply(lambda x: [np.array(x)], axis=1).apply(lambda x: x[0]))
    #    return X,Y,Z#

    #def numpyConverter(X):
    #    if isinstance(X, pd.core.frame.DataFrame):
    #        n_windows = X.shape[0]
    #        n_samples = X.iloc[0,0].shape[0]
    #        Y = np.empty([n_windows,n_samples])
    #        for i in range(n_windows):
    #            Y[i,:] = X.iloc[i,0]
    #    return Y
    #def convertingtonumpy(X,Y=None,Z=None):
    #    X = numpyConverter(X)
    #    if Y is None:
    #        return X
    #    Y = numpyConverter(Y)
    #    if Z is None:
    #        return X,Y
    #    Z = numpyConverter(Z)
    #    return X,Y,Z

## Converts Pandas Dataframes to Numpy
def convertingtonumpy(x_pool, length = 100):

    size = x_pool.shape[0]
    numpyFormat = np.empty([size,length])    
    
    for i in range(size):
        for j in range(length):
            numpyFormat[i,j] = x_pool['dim_0'][i][j]


    return numpyFormat

## Converts UCR Datasets from http://www.timeseriesclassification.com/ to a useful format: \n
# windows = Numpy matrix: Every row is one window \n
# Labels = numpy array (1D)
def postProcessSetfromTSCcom(windows_pool,windows_test, y_pool, y_test,window_length):
    #http://www.timeseriesclassification.com/
    #####################################################################################
    # handling of stupid Panda format

    windows_pool = convertingtonumpy(windows_pool,length = window_length)
    windows_test = convertingtonumpy(windows_test,length = window_length)

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
def loadData(set_type): 
    ###################
    ## Data Generation
    ###################
    if set_type == "generator": 
        #Settings
        windowLength = 1
        samplesPerWindow = 100
        n_classes = 6
        n_windows = 1000
        Gen = Datagenerator2dontcare.DataGenerator2dontcare(n_samples= samplesPerWindow,resolution = windowLength/samplesPerWindow,SNR_dB = 50, variation = 1, n_classes=n_classes, useseed = False, seed = 5)
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
        windows_pool, windows_test, y_pool, y_test = postProcessSetfromTSCcom(windows_pool, windows_test, y_pool, y_test,samplesPerWindow)
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

## Test run that saves Confusion matrix, confidences and queried samples after every iteration or saves the time (depends on setting)
def Test_run(obj,windows_pool, windows_test, y_pool, y_test):
    start_test = time.time()
    if Dataset == "generator":
        windows_pool, windows_test, y_pool, y_test, class_names, windowLength, n_classes, samplesPerWindow = loadData(Dataset)
        print("Generating time: " + str(time.time() - start_test))

    ## Feature Space Transformation
    x_pool, x_test = obj.fs_transform(windows_pool), obj.fs_transform(windows_test) # Feature Space Transformation, comment out if obj.fast_mode and not obj.singleErrorOutput and Generator used 
    x_pool_ID = np.arange(x_pool.shape[0])

    #x_pool, x_test = obj.normalizefeatures(x_pool), obj.normalizefeatures(x_test) #features normalisieren
    fst_done = time.time() #Feature Space Transformation Done
    print("FST time: " + str(fst_done - start_test))


    #print(x_pool.shape)

    if obj.timing_run != True:
        y_pool_complete = np.copy(y_pool)

    ## Random Selection of first annotated labels: replace with GUI or Clustering
    initial_idx = np.random.choice(range(len(x_pool)), size=obj.n_initial, replace=False)    

    ## workaround for problem with less samples: At least 2 classes must exist
    while y_pool[initial_idx].max() == y_pool[initial_idx].min():
        initial_idx = np.random.choice(range(len(x_pool)), size=obj.n_initial, replace=False)    
        print("all same class, try again")

    


    x_initial, y_initial = x_pool[initial_idx], y_pool[initial_idx]

    x_pool, x_pool_ID, y_pool = np.delete(x_pool, initial_idx, axis=0), np.delete(x_pool_ID, initial_idx, axis=0), np.delete(y_pool, initial_idx, axis=0)


    if windows_pool is not None:
        windows_pool = np.delete(windows_pool,initial_idx, axis=0)

    if obj.timing_run == True:
        Times = obj.ActiveLearning_timing_run(x_initial,y_initial,x_pool,y_pool)
        Times = np.append(fst_done,Times)
        return Times

    else: 
        receivedlabels, ConfusionData, average_cert, average_cert_error, average_cert_true = obj.ActiveLearning(x_initial,y_initial,x_pool, x_pool_ID, y_pool,x_test,y_test, windows_pool, windows_test)
        return y_pool_complete,receivedlabels,ConfusionData, average_cert, average_cert_error, average_cert_true

## Slim Function for real application 
def slimloadData(Type = "generator"): 
    ## slim function to load dataset
    windows_pool, windows_test, y_pool, y_test, class_names, windowLength, n_classes, samplesPerWindow = loadData(Type)
    return windows_pool, y_pool, class_names, windowLength, n_classes 

## Dummy User owns the knowledge
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
    class_names,windowLength,user = None,None,None

    # create Storage
    labels = np.ones(windows_pool.shape[0])*-1
        # -1 is no label
        # 0 is first class, last is don't care
        # Names in list "class_names"

    print("Feature Space Transformation")
    # Create Learner Object
    Learner = ALtools.RealActiveLearner(windows_pool,n_classes,algo) 

    ######################################
    ## Begin
    ######################################

    #receive fist few labels
    if coldStart == True:
        first_window_IDs, first_labels  = GUI.user.givefirstLabels(50)

    #store labels
    labels[first_window_IDs] = first_labels

    #can be removed for BOSS start
    ## workaround for problem with less samples: At least 2 different classes must exist, e.g. one interesting class and don't care
    while first_labels.max() == first_labels.min(): #all the same class    
        print("all windows are the same class, try again")
        first_window_IDs, first_labels  = user.givefirstLabels(50)

    #inital training
    Learner.initialTraining(first_window_IDs,first_labels)

    #first Query
    windowIDs, predictedLabels = Learner.Realquery()

    dontCareCounter = 0
    exit = False


    ## Active Learning Process
    while not exit:
        #get label for windows
        windows = windows_pool[windowIDs]
        realLabels = GUI.getlabel(windows,predictedLabels,windowIDs)
        
    
        #store labels
        labels[windowIDs] = realLabels



        ## Output 
        print("predicted (top) and Real (bottom)")
        print(predictedLabels.astype(np.int64))
        time.sleep(1)
        print(realLabels.astype(np.int64))
        print("")

        

        ## Exit Criteria
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
## Example of Real Application
###################################
if False: # Set False for Testing


    ## load Data
    print("load Data")
    windows_pool, y_pool, class_names, windowLength, n_classes = slimloadData() # "InsectWingbeat"  "InsectWingbeat"
    user = Dummy_User(y_pool,class_names) # User has the knowledge
    y_pool = None

    #windows, labels = LabelingProcess(windows_pool, class_names, windowLength, n_classes, user)

    # start with BOSS
    windows, labels = LabelingProcess(windows_pool, class_names, windowLength, n_classes, user, algo = "BOSS", dontcareLimit= 10)
    
    # switch to ROCKET
    windows, labels = LabelingProcess(windows_pool, class_names, windowLength, n_classes, user, labels = labels) #, algo = "ROCKET", dontcareLimit= 50

    labels[np.where(labels == -1)] = n_classes -1  # label not labeled windows with "dont care"
    difference = np.where(labels != user.y_pool)
    print(labels[difference])
    print(user.y_pool[difference])

    exit()






###################################
#    -           __
#  --          ~( @\   \
# ---   _________]_[__/_>________
#      /  ____ \ <>     |  ____  \
#     =\_/ __ \_\_______|_/ __ \__D
# ________(__)_____________(__)____
###################################
# Test environment
###################################

showprogress = False
detailResults = False
singleErrorOutput = False
saveresults = True
savefig = False
fast_timing_run = False # Fast mode to measure proccesing speed


#saveALL = False
#saveSequence = True

n = 25 # n runs
if fast_timing_run == True and (saveresults or detailResults or singleErrorOutput or savefig or showprogress):
    print('\x1b[6;30;42m' + "SLOW fast mode !"+'\x1b[0m')
if detailResults and n !=1:
    print('\x1b[6;30;42m' + "more than one run with detailed output for every single one"+'\x1b[0m')
if singleErrorOutput and n!= 1:
    print('\x1b[6;30;42m'+ '!!!'+'\x1b[0m' +"more than one run with single ERROR output for every single one")

## TSC Algorithm
#algo = "Mine"
algo = "BOSS"
#algo = "BOSSVS"
#algo = "ROCKET"

## Query Strategies
#query = "random"
query = "certainty"
#query = "entropy"
#query = "uncertainty"
#query = "imbEntropy"
#query = "imbEntropy2"


## Dataset
#Dataset = "generator"
Dataset = "GunPoint"
#Dataset = "FaceAll"
#Dataset = "Crop"

bagsize = 1


## Data Generation
windows_pool, windows_test, y_pool, y_test, class_names, windowLength, n_classes, samplesPerWindow = loadData(Dataset)

## old Comments
    #n_classes + samplesPerWindow could be calculated afterwards: 
    #n_classes = np.max(y_pool.max(),y_test.max())
    #samplesPerWindow = windows_pool.shape()[1]

    #if obj.fast_mode and not obj.singleErrorOutput:       
    #    #x_pool, x_test, y_pool, y_test = obj.GeneratePool(obj.n_windows)
    #    train_windows = None
    #    test_windows = None
    #################
    #### GUI ########
    #################
    # User Interaction
    #self.class_names = class_names  #only used to decode class numbers in visualization
    #self.windowLength = windowLength # only used for scaling of x-axis in visualization


    #obj = obersteKlasse(Gen,n_windows = 1000,n_initial=100,n_queries=900,fast_mode = True, auto_annotation = True)
    #obj = ALtools.ActiveLearningTools(Generator = Gen,n_windows = 560,n_initial=100,n_queries=0,algo=algo,query=query,fast_mode = True, auto_annotation = True,query_bagsize = 10,timing_run= fast_timing_run, detailResults= detailResults, singleErrorOutput= singleErrorOutput)


    #clf = sktime.classifiers.shapelet_based.ShapeletTransformClassifier(time_contract_in_mins=0.5)
    #clf = sktime.classifiers.distance_based.ProximityForest(n_stump_evaluations=1)

    #x_pool, x_test, y_pool, y_test = obj.GeneratePool(obj.n_windows)

    #panda_temp = pd.DataFrame(x_pool)
    #x_pd = pd.DataFrame(panda_temp.apply(lambda x: [np.array(x)], axis=1).apply(lambda x: x[0]))
    #print("fitting classifier")
    #clf.fit(x_pd,y_pool)
    #print("testing classifier")
    #clf.score(x_test,y_test)

## Initialization
obj = ALtools.ActiveLearningTools(n_classes,class_names, windowLength, samplesPerWindow, n_windows = windows_pool.shape[0],n_initial=5,n_queries=0,algo=algo,query=query,fast_mode = True, auto_annotation = True,query_bagsize = bagsize,timing_run= fast_timing_run, detailResults= detailResults, singleErrorOutput= singleErrorOutput)

Data = np.empty([5,n,obj.n_queries+1])
pool,receivedlabels,ConfusionData,certainties,error_certainties,true_certainties = np.empty([n,obj.n_windows]),np.empty([n,obj.n_initial+obj.n_queries*obj.queryBagsize]),np.empty([n,obj.n_classes,obj.n_classes,obj.n_queries+1]),np.empty([n,obj.n_queries+1]),np.empty([n,obj.n_queries+1]),np.empty([n,obj.n_queries+1])
Times = np.empty([n,obj.n_queries+2])

## do n test runs and store the results accordingly
for i in range(0,n):
    print("")
    print(str(i+1)+ " of "+ str(n))
    Starttime = time.time()
    if fast_timing_run == True:
        Times[i] = Test_run(obj,windows_pool, windows_test, y_pool, y_test)
        Times[i] -= Starttime #1st Value = Feature Space Transformation Time
    else:
        pool[i], receivedlabels[i], ConfusionData[i],certainties[i],error_certainties[i],true_certainties[i] = Test_run(obj,windows_pool, windows_test, y_pool, y_test)
    Stoptime = time.time()
    if obj.timing_run == True:
        print("Duration: " + str(Stoptime-Starttime)+ " s" )
    obj.reInit()


#################
## Postprocessing 
#################
CompleteRunningTime = Stoptime - Starttime
print("Duration: " + str(CompleteRunningTime))

###############
# change storage folder for test runs with Generator !!
###############
if Dataset == "generator":
    ### Yes you have to do this by hand currently...
    folder = "50dB1var100init1000total"
else:
    folder = Dataset

base_path = os.path.join(os.path.dirname(__file__),"resultsAL/"+folder)
filename = obj.algo +str(n) +"runs_"+obj.query +"_"+str(obj.queryBagsize)+"bag" +".npy"


if obj.timing_run == True:
    #1st Value = Feature Space Transformation Time (+Data Generation)
    #Times = np.append(0,Times) 
    #Times = np.append(Times,completeTime)
    Timingfile = os.path.join(base_path,("Times_"+filename))
    np.save(Timingfile,Times)
    plt.plot(range(0,Times[0].shape[0]),Times[-1])
    plt.show()

else: 
    accuracy_scores = calculate_accuracy(ConfusionData)
    #Store Data
    samples_trained = range(obj.n_initial,obj.n_queries+obj.n_initial+1)

    Data[0][0] = samples_trained
    Data[1] = accuracy_scores
    Data[2] = certainties
    Data[3] = error_certainties
    Data[4] = true_certainties

    if obj.algo is None:
        obj.algo = "other"

    
    Datafile = os.path.join(base_path,filename)
    ConfDatafile = os.path.join(base_path,("Conf_"+filename))
    queryOrderfile = os.path.join(base_path,("Queries_"+filename))
    poolFile = os.path.join(base_path,("Pools_"+filename))
    imgfile = os.path.join(base_path,"result.svg")


    if saveresults == True:
        #TODO create log file including model information
        #if useNeuralNet:
            #create_keras_model().summary()
        np.save(Datafile,Data)
        np.save(ConfDatafile,ConfusionData)
        np.save(queryOrderfile,receivedlabels)
        np.save(poolFile,pool)
    if savefig or showprogress:
        accuracy_scores = accuracy_scores.mean(0)
        error_certainties = error_certainties.mean(0)
        true_certainties = true_certainties.mean(0)
        ## result Visualization
        with plt.style.context('seaborn-white'):
            plt.figure(figsize=(10, 5))
            plt.title('Performance of the classifier during the active learning')
            plt.plot(samples_trained, accuracy_scores, label = obj.algo + " Classifier")#str(learner.estimator))+str(learner.query_strategy)
            plt.scatter(samples_trained, accuracy_scores)
            plt.plot(samples_trained,true_certainties,label="True Pos Certainty")
            plt.plot(samples_trained,error_certainties,label = "Error Certainty")
            plt.xlabel('number of queries')
            plt.ylabel('accuracy/Certainty')
            plt.legend()
            plt.figtext(0.01,0.04,"initial samples: "+ str(obj.n_initial))
            plt.ylim((0,1))
    if savefig == True:
        plt.savefig(imgfile)   
    if showprogress == True:
        plt.show()

print("")
plt.close('all')
