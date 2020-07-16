## @package ActiveLearningTools
# Collection of several TSC-algos and query strategies \n
# provides many interchangeable modules \n
# lots of TryOuts

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview

import tensorflow as tf 
#3 Nvidia Errors hier
print("---BEGIN---")
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
import os

## Versuch einer Problembehebung
#import sklearn
#import sktime
#from sktime.classifiers.shapelet_based import ShapeletTransformClassifier
#import sktime.classifiers
#import sktime.classifiers.distance_based

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

## tooling
#import modAL
from modAL.models import ActiveLearner
from modAL.utils import multi_argmax
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.preprocessing import scale
from joblib import dump,load
from sklearn import metrics
#import multiprocessing as mp
#import pandas as pd


## Classifier
##ANN
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics as kerasmetrics #sklearn metrics already imported as metrics
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

##ROCKET
from sklearn import preprocessing
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression

#KNN
from sklearn.neighbors import KNeighborsClassifier

#RF
from sklearn.ensemble import RandomForestClassifier

#SVM
from sklearn import svm

#base for own stuff
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from modAL.utils.data import data_vstack

## Time Series Classification state of the art
from pyts.classification import BOSSVS
from pyts.transformation import BOSS
#from sktime.classifiers.dictionary_based.boss import BOSSEnsemble
#from sktime.classifiers.dictionary_based.boss import BOSSIndividual
#from sktime.classifiers.distance_based import KNeighborsTimeSeriesClassifier
#tslearn
from rocket_functions import generate_kernels, apply_kernels

#query strategies
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import uncertainty_sampling
from scipy.stats import entropy #imbalanced entropy sampling



##KNN classifier for BOSS \n
#Probably 1 Nearest Neighbour \n
#distance measure: Cosine similary or BOSS-Distance \n
# Lots of tries to accelerate the processing speed of the BOSS-distance \n
class BOSS_NN_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.n_neighbors = 1       
        self.only_new = False
        #self.mp_pool = mp.Pool(12)
        self.X_train = None
        self.Y_train = None
        #import time
    ## BOSS distance 1: apply along functions
    def BOSS_dist_one_pair(self,x, y): #calculate from one sample x to one sample y (both with n timestamps)
            dist = (x-y)**2
            dist[x == 0] = 0
            return dist.sum()
    ## BOSS distance 1: apply along functions
    def BOSS_dist_to_all_Y(self,x, Y):
        x = np.apply_along_axis(self.BOSS_dist_one_pair, 1, Y, x)
        return x
    ## BOSS distance 2: numpy matrix calculation
    def matrixcalcBossDistance(self,X,Y):
        Y_transpose = np.transpose(Y)
        distance = - 2*np.matmul(X,Y_transpose)
        hauptDiagonaleX = np.matmul(np.diag(np.diag(np.matmul(X,np.transpose(X)))),np.ones(distance.shape))
        distance = np.add(hauptDiagonaleX,distance)
        hauptDiagonaleY = np.matmul(np.ones(distance.shape),np.diag(np.diag(np.matmul(Y,Y_transpose))))
        distance = np.add(distance,hauptDiagonaleY)

        ## search for zeros and undo the calculation with them
        correction = np.zeros(distance.shape)
        for i,x_window in enumerate(X): #windows in b1
            tempY = np.copy(Y_transpose)
            nonzeros = x_window != 0
            tempY[nonzeros] = 0
            correction[i] = np.diag(np.matmul(Y,tempY))
        distance -= correction

        return distance
    ## BOSS distance classic: very slow but understandable
    def calcBossDistance(self,x,Y, i_abs):
        distance = np.zeros(Y.shape[0])
        for i,y_time in enumerate(Y):
            for sample_X, sample_Y in zip(x, y_time):
                if sample_X != 0: ## specialty of BOSS distance 
                    distance[i] += (sample_X-sample_Y)**2
        return (i_abs,distance)
    ## BOSS distance: used function
    def Bossdistance(self,X,Y):
        X, Y = check_pairwise_arrays(X, Y)
        #distance = np.apply_along_axis(self.BOSS_dist_to_all_Y, 1, X, Y)

        distance = self.matrixcalcBossDistance(X,Y)

        ##################
        # Multiprocessing        
        #multiproc_data = [self.mp_pool.apply_async(self.calcBossDistance,(X_time,Y,i)) for i,X_time in enumerate(X)]
        #self.mp_pool.close()
        #self.mp_pool.join()
        ##################
        # Ergebnisse auspacken
        #distance = np.zeros([X.shape[0],Y.shape[0]])
        #for arr in multiproc_data:
        #    i, test = arr.get()
        #    distance[i] = test
        return distance
    ## opposite of distance
    def Boss_similarity(self,X,Y): ##superslow
        distance = self.Bossdistance(X,self.X_train)
        similarity = distance- np.amax(distance)
        return similarity
    ## fit classifier, for KNN (= lazy learner) store training data
    def fit(self,X,Y):
        # Check that X and y have correct shape
        X, Y = check_X_y(X, Y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(Y)

        
        if self.only_new == True:
            if self.X_train is None:
                self.X_train = X
                self.Y_train = Y
            else:
                #self.X_train,self.Y_train = np.append(self.X_train,X,axis = 0), np.append(self.Y_train,Y,axis = 0) #insert
                self.X_train, self.Y_train = data_vstack((self.X_train, X)),data_vstack((self.Y_train, Y))
        else:
            self.X_train = X
            self.Y_train = Y
        #return self
    ## predict confidences of every class for X  
    def predict_proba(self,X):
        # Check if fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
       
        similarity = cosine_similarity(X,self.X_train)
        distance = None
        #distance = self.Bossdistance(X,self.X_train)
        #print(similarity.shape)
        #print(similarity.argmax(axis=1).shape)
         
        if self.n_neighbors == 1:
            if distance is None: 
                neighborclass = self.Y_train[similarity.argmax(axis = 1)]
            else:
                neighborclass = self.Y_train[distance.argmin(axis = 1)]      
            #100% for next Neighbor, all other 0 
            probas = to_categorical(neighborclass,num_classes=int(max(self.Y_train))+1) #+1 für die null 
        else:
            #neighborclasses = self.Y_train[multi_argmax(similarity,1)]
            nearest = multi_argmax(similarity,self.n_neighbors)
            print(nearest.shape)
            for series in similarity:
                nearest = multi_argmax(series,self.n_neighbors)
                print(nearest.shape)
            nearest = self.Y_train[nearest]
        return probas
    ## predict class of x_test
    def predict(self, x_test):
        # Check is fit had been called
        #check_is_fitted(self)
        # Input validation
        #x_test = check_array(x_test)
        proba_predictedlabels = self.predict_proba(x_test)
        predictedlabels = np.argmax(proba_predictedlabels, axis=1)
        return predictedlabels
    ## calculate score, e.g. Accuracy
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
        return acc
        #value = [] #create list
        #value.append(sklearn.metrics.accuracy_score(y_test,y_pred))
        #return value
    ## reduce Data to accelerate calculations for big dataset \n
    # e.g. call this function after every X iterations \n
    # deletes the trainig Data thats not useful anymore or creates representive training Data like BOSS VS \n
    def reduceData(self,pool):
        pass
        #self.X_train: Maximize Entropy
        #search for the 200 samples that are the most at the outside of the feature space.
        #10NN-approach ? : keep the 500 samples that have least neighbours of the same class.
        #labels of the pool that has 9 of 10 neighbors of the same class could be labeled automatically with that label.

        ##pseudoCode
        # load samples in pool with label (X( (e.g. max +1))
        # calculate Distance (cosine similarity) from every sample to every sample #prevent double calculations # 1 huge Matrix
        # get label of 10 closest of each sample
        # if 9 or more labels are the same and sample label = X then label it automatically
        # if label = not X and number of neighbors > minimum number of neighbors in that list and list >= 200 then store nummber of labels in an array of size 200 and remove the sample with the smallest value
            # 200 samples in that list at the and are the new self.X_train.
            #-----or---- lke BOSSVS: sample with 9 or more same class neighbors will be merged to one representative 

        ## General Idea of keeping a huge Matrix with all calculated distance Values ? 

## BOSS VS Implementation \n
# from pyts missed some functions which are added here \n
class BOSSVS_classifier(BOSSVS):
    def __init__(self):
        super().__init__()

    def fit(self,X,Y):
        # Check that X and y have correct shape
        X, Y = check_X_y(X, Y)
        # Store the classes seen during fit
        #self.classes_ = unique_labels(Y)


        super().fit(X, Y)

    def predict_proba(self,X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        predicted = self.predict(X)

        probas = to_categorical(predicted,num_classes=self.tfidf_.shape[0])
        return probas

## Rocket Classifier \n
# probably Random Forest
class ROCKETClassifier(RandomForestClassifier):
    #svm.SVC
    #RandomForestClassifier
    #RidgeClassifier
    #RidgeClassifierCV
    #LogisticRegression
    def __init__(self):
        #super().__init__(alphas = np.logspace(-3, 3, 10), normalize = True) #ridgeclassifierCV
        #super().__init__(normalize=True) #ridgeclassifier 
        #super().__init__(max_iter = 200) #solver = ‘newton-cg’ # max_iter = 200 #Logistic Regression
        #super().__init__(probability=True) #SVM
        super().__init__(n_estimators=100, max_depth=10, n_jobs = -1) # Random Forest #warm_start n_estimators=100, max_depth=10, 
        #super().__init__(n_estimators=10) # Random Forest

    # Ridge
    #def predict_proba(self,X):
    #   d = self.decision_function(X)
    #   probs = np.exp(d)/(1+np.exp(d))
    #   probs = minmax_scale(d)
    #   return probs

    def __str__(self):
        return super().__str__()

## Tool and function Collection
class ActiveLearningTools():
    #######################
    #######################
    #  Settings
    #######################
    #######################
    #Most simple data: 
    #min 1000 for NN 
    # 10 for RF
    # 40 for SVM

    #classifier = KNeighborsClassifier(n_neighbors=5)
    #classifier = RandomForestClassifier(n_estimators=100,max_depth= 10, max_leaf_nodes= 10)
    #classifier = RandomForestClassifier()
    #classifier = svm.SVC(probability=True, decision_function_shape='ovo',gamma='scale',kernel='rbf') #doesn't work with very simple data
    #classifier = svm.SVC(probability=True, decision_function_shape='ovo' ,gamma='auto',kernel='rbf')
    #classifier = svm.SVC(probability=True, decision_function_shape='ovr', break_ties=True)
    #classifier = KerasClassifier(create_keras_model)

    #classifier = RandomForestClassifier()
    #classifier = KerasClassifier(self.create_keras_model)
    #classifier = svm.SVC(probability=True,decision_function_shape='ovr',kernel='rbf',gamma='scale',break_ties=True)

    def __init__(self, n_classes, class_names, windowLength, samplesPerWindow, n_windows = 10000, n_initial = 4*1000, n_queries = 0, algo= "BOSS", query = "entropy", fast_mode = True, auto_annotation = True,query_bagsize = 1,timing_run = False,detailResults = False, singleErrorOutput = False,  maxmaxvalue = 1e3):
        'Initialization'

        ## TSC Algorrihms
        self.algo = algo
    
        ## query strategy
        self.query = query
       
        ## bagsize
        self.queryBagsize = query_bagsize

        ## Data dimensions
        self.n_classes = n_classes
        self.samplesPerWindow = samplesPerWindow 
        self.n_windows = n_windows
        self.n_initial = n_initial
        if n_queries == 0:
            self.n_queries = int((n_windows-n_initial)/query_bagsize)
        else:
            self.n_queries = n_queries
        
        # User Interface values
        self.class_names = class_names  #only used to decode class numbers in visualization
        self.windowLength = windowLength # only used for scaling of x-axis in visualization
 
        ##user interaction Settings
        self.fast_mode = fast_mode
        self.auto_annotation = auto_annotation
        self.timing_run = timing_run
        self.detailResults = detailResults
        self.singleErrorOutput = singleErrorOutput
        if self.fast_mode and not self.auto_annotation:
            print("no manual annotation possible in fast mode")
            self.auto_annotation = True

        ## query Strategy Management
        if self.query == "random":
            self.query_strategy = self.random_sampling
        elif self.query == "certainty":
            self.query_strategy = self.imbalance_certainty_sampling
        elif self.query == "entropy":
            self.query_strategy = entropy_sampling
        elif self.query == "imbEntropy":
            self.query_strategy = self.imbalance_entropy_sampling
        elif self.query == "imbEntropy2":
            self.query_strategy = self.imbalance_entropy_sampling2    
        elif self.query == "uncertainty":
            self.query_strategy = uncertainty_sampling
        #self.query_strategy = self.imbalance_margin_sampling
        #self.query_strategy = self.certainty_sampling
        #self.query_strategy = self.imbalance_certainty_sampling
        #self.query_strategy = self.middlecertainty_sampling


        ## Algo Management
        if self.algo == "Mine":
            ## shape clf for preprocessing     
            self.base_path = os.path.join(os.path.dirname(__file__),"results/")
            self.shape_clf_file = os.path.join(self.base_path,"svm_80_20_clean_20db.joblib")
            self.shape_classifier = load(self.shape_clf_file)
            self.fs_transform = self.preprocessing_my_algo

            ## resample windows to a size of 100 (or retrain shape classifier for those number of samples)
            #if length < 100 : upsample 
            # upsampled = series.resample('D')
            #interpolated = upsampled.interpolate(method='linear')

            #elif length > 100: downsample
            #resample = series.resample('Q')
            #quarterly_mean_sales = resample.mean()


            ## classifier SVM
            self.classifier = svm.SVC(probability=True,decision_function_shape='ovr',kernel='rbf',gamma='scale',break_ties=True)


        if self.algo == "ROCKET":
            self.n_kernels = 10000 # 2 features per kernel 
            self.fs_transform = self.Rocket_transform
            self.classifier = ROCKETClassifier()
            ## C code generation with numba
            _ = generate_kernels(int(self.samplesPerWindow),int(self.n_kernels))
            zeros = np.zeros([self.n_windows,self.samplesPerWindow],dtype = float)
            _ = apply_kernels(np.zeros_like(zeros)[:, 1:], _)

            #self.classifier = LogisticRegression()
        
        if self.algo == "BOSS":
            self.fs_transform = self.BOSS_transform
            self.Boss = BOSS(word_size=2, n_bins=4, window_size=12, sparse=False)
            #self.Boss = BOSS(word_size=4, n_bins=2, window_size=10, sparse=False)
            self.classifier = BOSS_NN_classifier()
            #self.classifier = RandomForestClassifier()
        else:
            self.Boss = None

        if self.algo == "BOSSVS":
            self.fs_transform = self.identity
            self.BossVS = BOSSVS()#word_size=2, n_bins=3, window_size=10)
            self.classifier = BOSSVS_classifier()
        else:
            self.BossVS = None
        
        #self.fs_transform = self.identity
        #self.fs_transform = self.preprocessing_my_algo
        
        #self.classifier = RandomForestClassifier(n_estimators=100,max_depth= 10, max_leaf_nodes= 10)
        #self.classifier = KerasClassifier(self.create_keras_model)
        #self.classifier = svm.SVC(probability=True,decision_function_shape='ovr',kernel='rbf',gamma='scale',break_ties=True)
        #self.classifier = BOSSEnsemble(max_ensemble_size = 1) #sktime.classifiers.dictionary_based.BOSSIndividual
        #self.classifier = BOSSIndividual(window_size= 50, word_length= 6, alphabet_size = 4, norm = False )
        #self.classifier = BOSSVS()#word_size=2, n_bins=3, window_size=10)
        #self.classifier = sktime.classifiers.distance_based.ProximityForest(n_stump_evaluations=1)
        #self.classifier = KNeighborsTimeSeriesClassifier(metric="dtw")

        #special behaviour for ANNs    
        if self.classifier.__str__().split('0x')[0] == "<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at ":
            self.useNeuralNet = True
        else:
            self.useNeuralNet =  False

        # other
        self.maxmaxvalue = maxmaxvalue
        self.n_features = 10
    

    ### Functions

    ## resets everything
    def reInit(self):
        self.__init__(n_classes = self.n_classes ,class_names = self.class_names, windowLength = self.windowLength, samplesPerWindow = self.samplesPerWindow, n_windows = self.n_windows, n_initial = self.n_initial, n_queries = self.n_queries, algo= self.algo, query = self.query, fast_mode = self.fast_mode, auto_annotation = self.auto_annotation,query_bagsize = self.queryBagsize,timing_run= self.timing_run, maxmaxvalue = self.maxmaxvalue)


    #Query strategies

    ## No Active Learning
    def random_sampling(self,classifier,X_pool):
        n_samples = len(X_pool)
        query_idx = np.random.choice(range(n_samples),size = self.queryBagsize, replace=False)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(self.queryBagsize,-1)

    ## Certainty Sampling \n
    # Ununcertainty \n
    # max Confidence 
    def certainty_sampling(self,classifier,X_pool):
        probas = self.classifier.predict_proba(X_pool)
        probas = np.max(probas,axis = 1)
        query_idx = multi_argmax(probas,n_instances= self.queryBagsize)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(self.queryBagsize,-1)
   
    def imbalance_certainty_sampling(self,classifier,X_pool): ## last class is the "don't care" class and gets cut off
        probas = self.classifier.predict_proba(X_pool)
        probas = probas[:,0:-1] #cut off "don't cares"
        probas = np.max(probas,axis = 1)
        query_idx = multi_argmax(probas,n_instances= self.queryBagsize)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(self.queryBagsize,-1)

    def imbalance_entropy_sampling(self,classifier,X_pool): ## last class is the "don't care" class and gets cut off      
        probas = self.classifier.predict_proba(X_pool)
        probas = probas[:,0:-1] #cut off "don't cares"
        calculated_entropy =  np.transpose(entropy(np.transpose(probas)))
        query_idx = multi_argmax(calculated_entropy, n_instances=self.queryBagsize)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(self.queryBagsize,-1)

    def imbalance_entropy_sampling2(self,classifier,X_pool): ## last class is the "don't care" class and gets cut off      
        cert_weight = int(10) # must be 1 or higher : The higher it is the more weight is on entropy and less on certainty,
        probas = self.classifier.predict_proba(X_pool)
        
        probas_cut = probas[:,0:-1] #cut off "don't cares"

        max_probas_cut = np.max(probas_cut,axis = 1)    #similar to certaintysampling.
        if X_pool.shape[0] < cert_weight*self.queryBagsize: # At the end are not more samples in the pool.
            cert_weight = 1
        query_idx = multi_argmax(max_probas_cut,n_instances= self.queryBagsize*cert_weight) # certainty weights times that many closest samples
        # Of this smaller pool the samples with the highest entropy get choosen
        calculated_entropy =  np.transpose(entropy(np.transpose(probas[query_idx]))) # calc entropy of all probs ( or only of the important classes with probas_cut)
        query_idx_idx = multi_argmax(calculated_entropy,n_instances=self.queryBagsize)  #choose highest entropy
        query_idx = query_idx[query_idx_idx] # pick from the smaller pool
        return query_idx.reshape(-1), X_pool[query_idx].reshape(self.queryBagsize,-1)

    def imbalance_margin_sampling(self,classifier,X_pool): ## last class is the "don't care" class and gets cut off
        from modAL.utils.selection import multi_argmax
        probas = self.classifier.predict_proba(X_pool)
        probas = probas[:,0:-1] #cut off "don't cares"
        if probas.shape[1] == 1:
            return np.zeros(shape=(probas.shape[0],))
        part = np.partition(-probas, 1, axis=1)
        margin = part[:, 0] - part[:, 1]
        query_idx = multi_argmax(margin, n_instances=1)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(1,-1)
    
    def middlecertainty_sampling(self,classifier,X_pool):
        probas = abs(0.1-self.classifier.predict_proba(X_pool))
        probas = np.min(probas,axis = 1)
        query_idx = np.argmin(probas)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(1,-1)

    #special Classifier
    def create_keras_model(self): 
        """
        This function compiles and returns a Keras model.
        Should be passed to KerasClassifier in the Keras scikit-learn API.
        """
        opt = optimizers.Adam(lr=10**(-1*4), beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
        #opt = optimizers.SGD(lr=0.00001, momentum=0.0, decay=0.0, nesterov=False)
        loss = losses.categorical_crossentropy
        def top_3_categorical_accuracy(y_true, y_pred):
            return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
        
        #TODO Netz anpassen 
        #10 lstms
        #1d convolutional
        #dense am schnellsten (sinnvoll?)
        #GRU


        ## Variante 1 : Alfs 
        model = Sequential()
        model.add(layers.LSTM(units=100, use_bias=False, return_sequences=True, input_shape=(1,self.n_features)))
        #model.add(layers.LSTM(250, return_sequences=True))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        #model.add(layers.Dense(100))
        model.add(layers.Dense(self.n_classes))
        model.add(layers.Activation("softmax"))
        #model.summary()
        model.compile(  optimizer = opt,
                loss = loss,
                #metrics = [metrics.categorical_accuracy, top_3_categorical_accuracy]
                metrics=["accuracy"]
                )

        ##Variante 2 : Igors Keras example
        #model = Sequential()
        #model.add(Embedding(max_features, 128, input_length=maxlen))
        #model.add(Bidirectional(LSTM(64)))
        #model.add(Dropout(0.5))
        #model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        #model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])   
        #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        ## Variante 3 : Akif Cinar Acc. Train/test 0.79/0.64 mit 120825 Windows
        #model = Sequential()
        #model.add(layers.LSTM(units = 70, use_bias = False, return_sequences =  True, input_shape = (training_generator.getTimeSeriesLength(), training_generator.getNumberOfFeatures())))
        #model.add(layers.Dropout(0.5))
        #model.add(layers.Flatten())
        #model.add(layers.Dense(training_generator.getNumberOfLabels(), activation = 'softmax'))

        #Variante 4: Inception | Alexnet for time sereis
        return model

    #Feature space transformations
    def preprocessing_my_algo(self,windows):
        n_windows = windows.shape[0]
        features = np.empty([n_windows ,10])
        i = 0
        for window in windows:
            window = np.clip(window,-self.maxmaxvalue,self.maxmaxvalue)
            mean = np.mean(window)
            mmrange = max(window)-min(window)
            features[i,0:8] = self.shape_classifier.predict_proba(minmax_scale(window,feature_range=(0,1),axis = 0).reshape(1,-1))[0] # shape numbers
            features[i,8] = mean
            features[i,9] = mmrange
            i+=1
        features = self.normalizefeatures(features)
        return features

    def Rocket_transform(self, windows):
        #normalized to have a mean of zero and a standard deviation of one
        #windows = (windows - windows.mean(axis =1,keepdims=True)) / (windows.std(axis =1,keepdims = True) + 1e-8)  
        kernels = generate_kernels(windows.shape[1],self.n_kernels)

        ## C code generation with numba
        # wird normalerweise übersprungnen da C Code bereits im Konstruktor generiert wird
        _ = generate_kernels(int(self.samplesPerWindow),int(self.n_kernels))
        zeros = np.zeros([self.n_windows,self.samplesPerWindow],dtype = float)
        _ = apply_kernels(np.zeros_like(zeros)[:, 1:], _)
        # apply c code
        features = apply_kernels(windows,kernels)
        ##Scaling for Logistic Regression
        #for feature in features:
        #    feature = preprocessing.scale(feature)
        return features

    def identity(self,windows):
        return windows #no transformation

    def BOSS_transform(self, windows):    
        X_boss = self.Boss.fit_transform(windows)

        ## General Idea of keeping a huge Matrix with all calculated distance Values ? 

        return X_boss
   
    #Visualizations
    def calculate_certs(self,y_test, predictedlabels, proba_predictedlabels):
        tempn_windows = y_test.shape[0]
        #av. cert.
        average_certainty= 0
        for i in range(tempn_windows):
            average_certainty += np.max(proba_predictedlabels[i])
        average_certainty = average_certainty/tempn_windows

        #av error cert. & av correct cert.
        average_certainty_error = 0
        average_certainty_true = 0
        errorcount = 0
        for i in range(tempn_windows):
            if y_test[i] != predictedlabels[i]:
                average_certainty_error +=np.max(proba_predictedlabels[i])
                errorcount +=1
            elif y_test[i] == predictedlabels[i]:
                average_certainty_true +=np.max(proba_predictedlabels[i])
        if average_certainty_error != 0:
            average_certainty_error = average_certainty_error/errorcount
        if average_certainty_true != 0:
            average_certainty_true = average_certainty_true/(tempn_windows-errorcount)

        return average_certainty, average_certainty_error, average_certainty_true

    def visualize_certs(self,y_test,predictedlabels, proba_predictedlabels):
        tempn_windows = y_test.shape[0]
        True_samples = np.empty(0)
        False_samples = np.empty(0)
        for i in range(tempn_windows):
            if y_test[i] == predictedlabels[i]:
                True_samples = np.append(True_samples,np.max(proba_predictedlabels[i]))
            elif y_test[i] != predictedlabels[i]:
                False_samples = np.append(False_samples,np.max(proba_predictedlabels[i]))

        plt.scatter(range(True_samples.shape[0]),True_samples,color = 'green')
        plt.scatter(range(False_samples.shape[0]),False_samples,color = 'red')
        plt.show()
        return 

    def single_error_visualizaion(self,real,predicted,probas,features,window,Datagen):
        with plt.style.context('seaborn-white'):
            fig = plt.figure(constrained_layout = True)
            gs = fig.add_gridspec(10,10)
            
            fig1 = fig.add_subplot(gs[:,:4])
            plt.title("real:\n" + Datagen.class_names[real] + " with "+ str(np.around(probas[real]*100,2))+ " %")
            plt.scatter(np.linspace(0,Datagen.windowsize,Datagen.n_samples),window)
            plt.plot(np.linspace(0,Datagen.windowsize,Datagen.n_samples),window,linewidth = 0.5,linestyle = 'dashed')
            #plt.axhline(y = np.mean(windows[i]),color ='#2ca02c')
            #plt.axhline(y = max(windows[i]),color = '#d62728')
            #plt.axhline(y = min(windows[i]),color = '#d62728')

            fig2 = fig.add_subplot(gs[:,4:9],polar =True)
            plt.title("predicted:\n" + Datagen.class_names[predicted]+ " with "+ str(np.around(probas[predicted]*100,2))+ " %")
            shapes = np.empty(9)
            shapes[:8] = features[:8]
            shapes[8] = shapes[0]
            categories = ['const', 'rise', 'fall', 'sine', 'dirac up', 'dirac down', 'step up', 'step down']
            angles = [n / float(8) * 2 * np.pi for n in range(8)]
            angles += angles[:1]

            # Draw one axe per variable + add labels
            plt.xticks(angles[:-1], categories, color='grey', size=8)
            fig2.set_rlabel_position(0)
            plt.yticks([0.1,0.5,0.9], [".1",".5",".9"], color="grey", size=7)
            plt.ylim(0,1)

            plt.plot(angles, shapes, linewidth=1, linestyle='solid')
            plt.fill(angles, shapes, 'b', alpha=0.1)

            fig3 = fig.add_subplot(gs[:5,9])
            plt.title("normalized \n mean")
            plt.plot(np.linspace(0,1,10),np.ones(10)*features[8],scaley=False)
            fig3.set_xticklabels([])

            fig4 = fig.add_subplot(gs[5:,9])
            plt.title("normalized \n range")
            plt.plot(np.linspace(0,1,10),np.ones(10)*features[9],scaley=False)
            fig4.set_xticklabels([])


            print(features)
            #print(np.round(proba_predictedlabels[i],2))
            #plt.draw()
            #plt.pause(0.01)
            plt.show()
    
    def calcscore(self,learner,x_test,y_test):
        y_pred = learner.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
        return acc

    def visualizeBoss(self,X_boss,y_train):
        # Visualize the transformation for the first time series
        boss = obj.Boss

        plt.figure(figsize=(6, 4))
        plt.clf()
        vocabulary_length = len(boss.vocabulary_)
        width = 0.5
        n_classes = obj.n_classes
        for i in range(n_classes):
            plt.bar(np.arange(vocabulary_length) - (width/n_classes)*(n_classes/2) + (width/n_classes)*i, X_boss[y_train == i][0],
                width=width/(n_classes), label='First time series in class '+obj.class_names[i])     
            #print(X_boss[y_train == i][0])
        plt.xticks(np.arange(vocabulary_length),
                np.vectorize(boss.vocabulary_.get)(np.arange(X_boss[0].size)),
                fontsize=12)
        y_max = np.max(np.concatenate([X_boss[y_train == 1][0],
                                    X_boss[y_train == 2][0]]))
        plt.yticks(np.arange(y_max + 1), fontsize=12)
        plt.xlabel("Words", fontsize=14)
        plt.ylabel("Frequencies", fontsize=14)
        plt.title("BOSS transformation", fontsize=16)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.show()

    #Data handling
    def cutinwindows(self,timeseries):
        #n_samples = self.samplesPerWindow
        n_samples = self.n_windows
        Windowlength = self.samplesPerWindow
        window = np.empty([n_samples,Windowlength])
        mean = np.empty(n_samples)
        for i in range(n_samples):
            for j in range(Windowlength):
                window[i][j]= timeseries[i*Windowlength+j]
            mean[i] = np.mean(window[i][0:Windowlength])
            Range = max(window[i][0:Windowlength])-min(window[i][0:Windowlength])
            
            
            
            window[i]= minmax_scale(window[i],feature_range=(0,1),axis = 0)

        mean = minmax_scale(mean,feature_range=(0,1),axis = 0)
        #print(mean)
        #plt.plot(timestamps,timeseries)
        #plt.show()
        return window


    """
    def GeneratePool(self,n_windows):
        ""
        generates Pool of size <n_windows> and additional training data of size  <n_windows*test_size>
        ""
        test_size = 0.2

        n_test_windows = int(n_windows*(test_size/(1-test_size)))
        n_total_windows = n_windows + n_test_windows
        #X = np.empty([n_total_windows,10])
        X = np.empty([n_total_windows,self.samplesPerWindow])
        Y = np.empty(n_total_windows)

        i= 0
        for window,label in self.Gen.genWindow(n_total_windows): #n windows generieren  
            X[i] = window
            Y[i] = label
            i+=1
            print(str(np.round((i/(n_total_windows))*100,3)) + " %" + " generiert", end = ' ')
            print("\r", end='')

        test = time.time()
        X = self.fs_transform(X) # features berechnen
        #X = self.normalizefeatures(X) #features normalisieren
        if self.timing_run == True:
            print("FST: " +str(time.time() - test)+ " s")
        #print("")

        #reshaping for NN
        if self.useNeuralNet == True:
            X = np.reshape(X, (X.shape[0],1,X.shape[1]))
            Y = np.reshape(Y, (Y.shape[0], 1))
            Y = to_categorical(Y,num_classes=self.n_classes)

        # Split
        X_pool, X_test, Y_pool, Y_test = train_test_split(X,Y,test_size=test_size)

        return X_pool, X_test, Y_pool, Y_test
   
    def GeneratePool_with_windows(self,n_windows):
        ""
        generates Pool of size <n_windows> and additional training data of size  <n_windows*test_size>
        and saves windows
        ""
        test_size = 0.2

        n_test_windows = int(n_windows*(test_size/(1-test_size)))
        n_total_windows = n_windows + n_test_windows
        X = np.empty([n_total_windows,10])
        Y = np.empty(n_total_windows)
        windows = np.empty([n_total_windows,self.samplesPerWindow])
        i= 0
        for window,label in .genWindow(n_total_windows): #n windows generieren 
            windows[i,:] = window
            Y[i] = label
            i+=1
            print(str(np.round((i/(n_total_windows))*100,3)) + " %" + " generiert", end = ' ')
            print("\r", end='')

        X = self.fs_transform(windows) # features berechnen 
        X = self.normalizefeatures(X) #features normalisieren

        #reshaping for NN
        if self.useNeuralNet == True:
            X = np.reshape(X, (X.shape[0],1,X.shape[1]))
            Y = np.reshape(Y, (Y.shape[0], 1))
            Y = to_categorical(Y,num_classes=self.n_classes)

        # Split
        X_pool, X_test, Y_pool, Y_test, train_windows, test_windows = train_test_split(X,Y,windows,test_size=test_size)

        return X_pool, X_test, Y_pool, Y_test, train_windows, test_windows
    """


    #main 
    def ActiveLearning(self,x_labeled,y_labeled,x_pool,x_pool_ID,y_pool,x_test,y_test, train_windows = None, test_windows = None):
        ## Active Learning

        #x_pool, x_test, x_labeled = convertingtopanda(x_pool,x_test,x_labeled)
        
        learner = ActiveLearner(
            estimator=self.classifier,
            query_strategy=self.query_strategy,
            X_training=x_labeled, y_training=y_labeled
        )

        accuracy_scores = [learner.score(x_test, y_test)]
        #if score ist not available :
        #accuracy_scores = self.calcscore(learner,x_test,y_test)
        #print(accuracy_scores)
        #print(type(accuracy_scores))
        classifier = learner.estimator

        proba_predictedlabels = classifier.predict_proba(x_test)
        predictedlabels = np.argmax(proba_predictedlabels, axis=1)
        average_cert, average_cert_error, average_cert_true = self.calculate_certs(y_test, predictedlabels, proba_predictedlabels)

        #if saveALL == True:

        

        conf_y_test = np.copy(y_test)
        conf_y_pred = np.copy(predictedlabels)
        for i in range(self.n_classes):
            conf_y_test = np.append(conf_y_test,i)
            conf_y_pred = np.append(conf_y_pred,i)
        ConfusionData = metrics.confusion_matrix(conf_y_test, conf_y_pred) - np.eye(self.n_classes) #Confusionmatrix has fixed size even when not every label is in Dataset yet

        #if saveSequence == True:
        #    print(learner.y_training)
        #######################
        # active Learning
        #######################
    
        windowexists = False
        for i in range(self.n_queries):
            # find instance with query strategy
            query_idx, query_inst = learner.query(x_pool)
            #print(query_idx)
            trueLabel = y_pool[query_idx]

            if not self.fast_mode: #show instance
                with plt.style.context('seaborn-white'):
                    if windowexists == False:
                        plt.ion()
                        plt.figure(figsize=(10, 5))
                        fig1 = plt.subplot(1, 2, 1)
                        plt.title('window to label')      
                    fig1.cla()
                    if self.useNeuralNet == True:
                        show_inst = query_inst[0][0]
                    else:
                        show_inst = query_inst[0]
                    #plt.plot(np.linspace(self.Gen.resolution, self.Gen.windowsize, self.n_features),show_inst)
                    #plt.scatter(np.linspace(self.Gen.resolution, self.Gen.windowsize, self.n_features),show_inst)
                    fig1.scatter(np.linspace(0,self.windowLength,self.samplesPerWindow),train_windows[query_idx,:][0])
                    fig1.plot(np.linspace(0,self.windowLength,self.samplesPerWindow),train_windows[query_idx,:][0],linewidth = 0.5,linestyle = 'dashed')           
                    if windowexists == False:
                        fig2 = plt.subplot(1, 2, 2)
                        plt.title('Accuracy of your model')
                    fig2.plot(range(i+1), accuracy_scores)
                    fig2.scatter(range(i+1), accuracy_scores)
                    if windowexists == False:
                        plt.xlabel('number of queries')
                        plt.ylabel('accuracy')
                    plt.draw()
                    plt.pause(0.01)
                windowexists = True

                for j in range(0,len(self.class_names)):
                    print(str(j) +" " +self.class_names[j])    
                print("What is this?")
                if self.auto_annotation: #oracle = machine
                    #time.sleep(0.5)
                    if self.useNeuralNet:
                        print(np.argmax(trueLabel))
                    else:
                        print(trueLabel)
                    #time.sleep(1)
            if self.fast_mode:
                print("question " + str(i+1) + " of " + str(self.n_queries)+ " | Accuracy: " + str(accuracy_scores[-1]) , end = ' ')
                if not self.useNeuralNet:
                    print("\r", end='')
            if self.auto_annotation: #oracle = machine
                #print("ID: " +str(x_pool_ID[query_idx]))
                y_new = np.array(trueLabel, dtype=int)
            elif not self.auto_annotation: #oracle = user
                ## GUI 
                answer = int(input())
                if self.useNeuralNet:
                    answer = to_categorical(answer,num_classes=self.n_classes)
                y_new = np.array([answer], dtype=int)

            learner.teach(query_inst,y_new,only_new=False) 
            #x_labeled, y_labeled = np.append(x_labeled,query_inst,axis = 0), np.append(y_labeled,trueLabel,axis = 0) #insert
            #print(learner.X_training.shape)
            x_pool, x_pool_ID, y_pool = np.delete(x_pool, query_idx, axis=0), np.delete(x_pool_ID, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)

            if self.fast_mode != True:
                train_windows = np.delete(train_windows,query_idx, axis=0)

            #x_test = convertingtopanda(x_test)
            #learner.X_training = convertingtopanda(learner.X_training)
            accuracy_scores.append(learner.score(x_test, y_test))
            #if score ist not available :
            #print(accuracy_scores)
            #accuracy_scores.append(self.calcscore(learner,x_test,y_test))

            ## calc for analysis from here 
            #tempn_windows = int(self.n_windows/10)
            classifier = learner.estimator
            proba_predictedlabels = classifier.predict_proba(x_test)

            predictedlabels = np.argmax(proba_predictedlabels, axis=1)

            average_certainty, average_certainty_error, average_certainty_true = self.calculate_certs(y_test, predictedlabels, proba_predictedlabels)
            #if saveALL == True:
            conf_y_test = np.copy(y_test)
            conf_y_pred = np.copy(predictedlabels)
            for i in range(self.n_classes):
                conf_y_test = np.append(conf_y_test,i)
                conf_y_pred = np.append(conf_y_pred,i)
            Confusionmatrix = metrics.confusion_matrix(conf_y_test, conf_y_pred) - np.eye(self.n_classes)
            ConfusionData = np.dstack((ConfusionData,Confusionmatrix))
                        
            #if saveSequence == True:
            #    print(learner.y_training)

            average_cert= np.append(average_cert,average_certainty)
            average_cert_error = np.append(average_cert_error,average_certainty_error)
            average_cert_true = np.append(average_cert_true,average_certainty_true)

        if self.detailResults or self.singleErrorOutput:
            plt.close('all')
            classifier = learner.estimator
            proba_predictedlabels = classifier.predict_proba(x_test)
            predictedlabels = np.argmax(proba_predictedlabels, axis=1)
            #predicted = classifier.predict(test_windows)

            if self.detailResults == True:
            #Confusion Matrix zur Auswertung
                if self.n_classes < 20:
                    signumbers = 2
                else:
                    signumbers = 1
                if self.useNeuralNet:
                    real = np.empty(y_test.shape[0])
                    for i in range(y_test.shape[0]):
                        real[i] = np.argmax(y_test[i])  
                    print("Classification report for classifier %s:\n%s\n"
                    % (classifier, metrics.classification_report(real, predictedlabels)))
                    #disp = metrics.plot_confusion_matrix(classifier, test_windows, real) #doesn't work with Keras
                else:
                    plt.close('all')
                    print("Classification report for classifier %s:\n%s\n"
                        % (classifier, metrics.classification_report(y_test, predictedlabels)))
                    disp = metrics.plot_confusion_matrix(classifier, x_test, y_test,normalize= 'true', display_labels= self.class_names , xticks_rotation= 'vertical', values_format='.'+str(signumbers)+'f', include_values=True) #labels = (0,1,2,3,4) #values_format='.1f' '.2g' 'true' #'pred' None
                    disp.figure_.set_size_inches(13,13)
                    #plt.savefig(imgfile)
                    #disp.figure_.suptitle("Confusion Matrix")
                    #print("Confusion matrix:\n%s" % disp.confusion_matrix)
                    plt.show()
                            
            if self.singleErrorOutput== True:
                #reshaping for NN
                #if self.useNeuralNet == True:
                #    train_windows= window[:1200,:] #windowandmean
                #    test_windows = window[1201:,:]
                for i in range(y_test.shape[0]):
                    if self.useNeuralNet:
                        #shape for NN
                        real = np.argmax(y_test[i]) # reverse one hot encoding
                    else:
                        #shape for other Classifiers
                        real = y_test[i]

                    predicted = predictedlabels[i]
                    #print(proba_predictedlabels[i])
                    if real != predicted:

                        self.single_error_visualizaion(int(y_test[i]),int(predicted),proba_predictedlabels[i],x_test[i],test_windows[i],self.Gen)

        #return np.asarray(accuracy_scores), average_cert, average_cert_error, average_cert_true
        return learner.y_training, ConfusionData, average_cert, average_cert_error, average_cert_true
   

    #Timing Run
    def ActiveLearning_timing_run(self,x_labeled,y_labeled,x_pool,y_pool):
        ## slim Active Learning
        learner = ActiveLearner(
            estimator=self.classifier,
            query_strategy=self.query_strategy,
            X_training=x_labeled, y_training=y_labeled
        )
        Times = time.time()


        windowexists = False

        deletethis = time.time()

        for i in range(self.n_queries):
            print("")
            print(str(time.time()-deletethis) + " seconds")
            # find instance with query strategy
            query_idx, query_inst = learner.query(x_pool)
            trueLabel = y_pool[query_idx]

            if not self.fast_mode: #show instance
                with plt.style.context('seaborn-white'):
                    if windowexists == False:
                        plt.ion()
                        plt.figure(figsize=(10, 5))
                        fig1 = plt.subplot(1, 2, 1)
                    if windowexists == False:
                        plt.title('window to label')      
                    fig1.cla()
                    if self.useNeuralNet == True:
                        show_inst = query_inst[0][0]
                    else:
                        show_inst = query_inst[0]
                    #plt.plot(np.linspace(self.Gen.resolution, self.Gen.windowsize, self.n_features),show_inst)
                    #plt.scatter(np.linspace(self.Gen.resolution, self.Gen.windowsize, self.n_features),show_inst)
                    fig1.scatter(np.linspace(0,self.windowLength,self.samplesPerWindow),train_windows[query_idx,:][0])
                    fig1.plot(np.linspace(0,self.windowLength,self.samplesPerWindow),train_windows[query_idx,:][0],linewidth = 0.5,linestyle = 'dashed')           
                    if windowexists == False:
                        fig2 = plt.subplot(1, 2, 2)
                        plt.title('Accuracy of your model')
                    fig2.plot(range(i+1), accuracy_scores)
                    fig2.scatter(range(i+1), accuracy_scores)
                    if windowexists == False:
                        plt.xlabel('number of queries')
                        plt.ylabel('accuracy')
                    plt.draw()
                    plt.pause(0.01)
                windowexists = True

                for j in range(0,len(self.class_names)):
                    print(str(j) +" " +self.class_names[j])    
                print("What is this?")
                if self.auto_annotation: #oracle = machine
                    #time.sleep(0.5)
                    if self.useNeuralNet:
                        print(np.argmax(trueLabel))
                    else:
                        print(trueLabel)
                    #time.sleep(1)

            if self.fast_mode:
                print("question " + str(i+1) + " of " + str(self.n_queries), end = ' ')
                if not self.useNeuralNet:
                    print("\r", end='')
            if self.auto_annotation: #oracle = machine
                y_new = np.array(trueLabel, dtype=int)
            elif not self.auto_annotation: #oracle = user
                answer = int(input())
                if self.useNeuralNet:
                    answer = to_categorical(answer,num_classes=self.n_classes)
                y_new = np.array([answer], dtype=int)


            learner.teach(query_inst,y_new,only_new=False) 
            #x_labeled, y_labeled = np.append(x_labeled,query_inst,axis = 0), np.append(y_labeled,trueLabel,axis = 0) #insert
            #print(learner.X_training.shape)
            x_pool, y_pool = np.delete(x_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
            if self.fast_mode != True:
                train_windows = np.delete(train_windows,query_idx, axis=0)


            Times = np.append(Times,time.time())


        return Times
   
    #others
    def normalizefeatures(self,feature):
        #normalize mean and range
        meanmax = 1e3
        rangemax = 1e2

        #mean
        feature[:,8] = np.clip(feature[:,8],-meanmax,meanmax)
        feature[:,8] = minmax_scale(feature[:,8],feature_range=(0,1),axis = 0)
        #range
        feature[:,9] = np.clip(feature[:,9],-rangemax,rangemax)
        feature[:,9] = minmax_scale(feature[:,9],feature_range=(0,1),axis = 0)
        return feature

## Uses only the choosen functions, settings and tools
class RealActiveLearner(ActiveLearningTools):
    ## init examples
        #    def __init__(self, n_classes, class_names, windowLength, samplesPerWindow, n_windows = 10000, n_initial = 4*1000, n_queries = 0, algo= "BOSS", query = "entropy", fast_mode = True, auto_annotation = True,query_bagsize = 1,timing_run = False,detailResults = False, singleErrorOutput = False,  maxmaxvalue = 1e3):
        #   obj = ALtools.ActiveLearningTools(n_classes,class_names, windowLength, samplesPerWindow, n_windows = windows_pool.shape[0],n_initial=5,n_queries=0,algo=algo,query=query,fast_mode = True, auto_annotation = True,query_bagsize = bagsize,timing_run= fast_timing_run, detailResults= detailResults, singleErrorOutput= singleErrorOutput)

    def __init__(self, windows_pool, n_classes,algo):
        n_windows = windows_pool.shape[0]
        samplesPerWindow = windows_pool.shape[1] #samples per window is need to accelerate ROCKET for faster FST with numba
        super().__init__(n_classes,class_names='', windowLength = 1, samplesPerWindow = samplesPerWindow, n_windows = n_windows,n_initial=0,n_queries=0,algo= algo,query="certainty",fast_mode = False, auto_annotation = False,query_bagsize = 10,timing_run= False, detailResults= False, singleErrorOutput= False)
        
        # Feature Space Transformation
        self.x_pool = self.fs_transform(windows_pool)
        
        # Give IDs to windows
        self.x_pool_ID = np.arange(self.x_pool.shape[0])

    def Realquery(self):
        query_idx, query_inst = self.learner.query(self.x_pool)
        self.current_queries = query_inst
        self.current_idx = query_idx
        windowsIDs = self.x_pool_ID[query_idx]

        proba_predictedlabels = self.learner.estimator.predict_proba(query_inst)
        predictedlabels = np.argmax(proba_predictedlabels, axis=1)
    
        return windowsIDs, predictedlabels

    def initialTraining(self, window_IDs, labels): 
        #poolIDs = window_IDs because inital training
        x_initial, y_initial = self.x_pool[window_IDs], labels
        self.learner = ActiveLearner(
            estimator=self.classifier,
            query_strategy=self.query_strategy,
            X_training=x_initial, y_training=y_initial
        )
        self.x_pool, self.x_pool_ID = np.delete(self.x_pool, window_IDs, axis=0), np.delete(self.x_pool_ID, window_IDs, axis=0)

    def ActiveLearningIteration(self,new_Labels):
        self.learner.teach(self.current_queries,new_Labels,only_new=False) 
        self.x_pool, self.x_pool_ID = np.delete(self.x_pool, self.current_idx, axis=0), np.delete(self.x_pool_ID, self.current_idx, axis=0)
        return self.Realquery()

    def change_query_strategy(self,newStrategy = "certainty"):
        self.query = newStrategy
        ## query Strategy Management
        if self.query == "random":
            self.query_strategy = self.random_sampling
        elif self.query == "certainty":
            self.query_strategy = self.imbalance_certainty_sampling
        elif self.query == "entropy":
            self.query_strategy = entropy_sampling
        elif self.query == "imbEntropy":
            self.query_strategy = self.imbalance_entropy_sampling
        elif self.query == "imbEntropy2":
            self.query_strategy = self.imbalance_entropy_sampling2    
        elif self.query == "uncertainty":
            self.query_strategy = uncertainty_sampling
        #self.query_strategy = self.imbalance_margin_sampling
        #self.query_strategy = self.certainty_sampling
        #self.query_strategy = self.imbalance_certainty_sampling
        #self.query_strategy = self.middlecertainty_sampling

        self.learner.query_strategy = self.query_strategy