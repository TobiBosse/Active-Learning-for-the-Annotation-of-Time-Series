## @package ActiveLearningTools
# Collection of choosen TSC-algos and query strategy \n

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview

# basics
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.utils import to_categorical
from modAL.utils import multi_argmax

# modAL
from modAL.models import ActiveLearner as modAL_ActiveLearner

# RF
from sklearn.ensemble import RandomForestClassifier

# KNN
from sklearn.metrics.pairwise import cosine_similarity

# Time Series Classification state of the art
from pyts.transformation import BOSS
from ActiveLearningEnv.rocket_functions import generate_kernels, apply_kernels

## Uses only the choosen functions, settings and tools
class ActiveLearner():

    ##KNN classifier for BOSS \n
    #Probably 1 Nearest Neighbour \n
    #distance measure: Cosine similarity
    class BOSS_NN_classifier(BaseEstimator, ClassifierMixin):
        def __init__(self):
            self.X_train = None
            self.Y_train = None

        ## fit classifier, for KNN (= lazy learner) store training data
        def fit(self,X,Y):
            # Check that X and y have correct shape
            X, Y = check_X_y(X, Y)
            # Store the classes seen during fit
            self.classes_ = unique_labels(Y)
            self.X_train = X
            self.Y_train = Y

        ## predict confidences of every class for X  
        def predict_proba(self,X):
            # Check if fit had been called
            check_is_fitted(self)
            # Input validation
            X = check_array(X)    
            similarity = cosine_similarity(X,self.X_train)
            neighborclass = self.Y_train[similarity.argmax(axis = 1)]
            probas = to_categorical(neighborclass,num_classes=int(max(self.Y_train))+1) #+1 für die null 
            return probas

        ## predict class of x_test
        def predict(self, x_test):
            proba_predictedlabels = self.predict_proba(x_test)
            predictedlabels = np.argmax(proba_predictedlabels, axis=1)
            return predictedlabels

    ## Rocket Classifier \n
    # probably Random Forest
    class ROCKETClassifier(RandomForestClassifier):
        def __init__(self):
            ## Settings for Random Forest
            super().__init__(n_estimators=100, max_depth=10, n_jobs = -1) # Random Forest #warm_start n_estimators=100, max_depth=10, 

        def __str__(self):
            return super().__str__()

    ## Query Strategy
    def imbalance_certainty_sampling(self,classifier,X_pool): ## last class is the "don't care" class and gets cut off
        probas = self.classifier.predict_proba(X_pool)
        probas = probas[:,0:-1] #cut off "don't cares"
        probas = np.max(probas,axis = 1)
        query_idx = multi_argmax(probas,n_instances= self.queryBagsize)
        return query_idx.reshape(-1), X_pool[query_idx].reshape(self.queryBagsize,-1)

    ## init function 
    def __init__(self, windows_pool, n_classes,algo):
        self.n_windows = windows_pool.shape[0]  # number fo windows
        self.samplesPerWindow = windows_pool.shape[1] #samples per window is need to accelerate ROCKET for faster FST with numba

        #super().__init__(n_classes,class_names='', windowLength = 1, samplesPerWindow = samplesPerWindow, n_windows = n_windows,n_initial=0,n_queries=0,algo= algo,query="certainty",fast_mode = False, auto_annotation = False,query_bagsize = 10,timing_run= False, detailResults= False, singleErrorOutput= False)
        
        ## TSC Algorrihms
        self.algo = algo
    
        ## query strategy
        self.query_strategy = self.imbalance_certainty_sampling
       
        ## bagsize setting
        self.queryBagsize = 10

        ##algo management and Settings
        if self.algo == "ROCKET":
            # number of kernels Setting
            self.n_kernels = 10000 # 2 features per kernel 
            self.fs_transform = self.Rocket_transform
            self.classifier = self.ROCKETClassifier()
            ## C code generation with numba
            _ = generate_kernels(int(self.samplesPerWindow),int(self.n_kernels))
            zeros = np.zeros([self.n_windows,self.samplesPerWindow],dtype = float)
            _ = apply_kernels(np.zeros_like(zeros)[:, 1:], _)

            #self.classifier = LogisticRegression()
        
        if self.algo == "BOSS":
            self.fs_transform = self.BOSS_transform
            # Boss Settings
            self.Boss = BOSS(word_size=2, n_bins=4, window_size=12, sparse=False)
            #self.Boss = BOSS(word_size=4, n_bins=2, window_size=10, sparse=False)
            self.classifier = self.BOSS_NN_classifier()
            #self.classifier = RandomForestClassifier()
        else:
            self.Boss = None


        # execute Feature Space Transformation
        self.x_pool = self.fs_transform(windows_pool)
        
        # Give IDs to windows
        self.x_pool_ID = np.arange(self.x_pool.shape[0])



    ## Feature Space Transformations

    ## BOSS FST
    def BOSS_transform(self, windows):    
        X_boss = self.Boss.fit_transform(windows)
        return X_boss

    ## ROCKET FST
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


    ## Real used functions

    ## search new interesting samples
    def query(self):
        # modal query
        query_idx, query_inst = self.learner.query(self.x_pool)
        ## queried samples 
        self.current_queries = query_inst
        ## and their position in Feature Space
        self.current_idx = query_idx
        ## window IDs = position in storage of windows
        windowsIDs = self.x_pool_ID[query_idx]

        proba_predictedlabels = self.learner.estimator.predict_proba(query_inst)
        predictedlabels = np.argmax(proba_predictedlabels, axis=1)
    
        return windowsIDs, predictedlabels

    ## initial training 
    def initialTraining(self, window_IDs, labels): 
        #poolIDs = window_IDs because inital training
        x_initial, y_initial = self.x_pool[window_IDs], labels
        self.learner = modAL_ActiveLearner(
            estimator=self.classifier,
            query_strategy=self.query_strategy,
            X_training=x_initial, y_training=y_initial
        )
        ## remove seen samples from pool
        self.x_pool, self.x_pool_ID = np.delete(self.x_pool, window_IDs, axis=0), np.delete(self.x_pool_ID, window_IDs, axis=0)

    ## whole Iteration as described in the software architecture "Realization"
    def ActiveLearningIteration(self,new_Labels):
        # learn new samples with new labels
        self.learner.teach(self.current_queries,new_Labels,only_new=False) 
        # remove learned samples from pool
        self.x_pool, self.x_pool_ID = np.delete(self.x_pool, self.current_idx, axis=0), np.delete(self.x_pool_ID, self.current_idx, axis=0)
        # query new ones
        return self.query()
