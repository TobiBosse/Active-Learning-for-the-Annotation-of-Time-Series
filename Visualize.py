## @package Visualize
# Function collection to visualize 

###############################
# In VS Code: Press CTRL + K + 0 (zero) to get an Overview

import numpy as np 
import matplotlib.pyplot as plt 
import os
import tikzplotlib
#import plotly

#import seaborn as sns

def getlabels():
    ## Data Generator for labels
    import Datagenerator2
    import Datagenerator2dontcare


    ## Init Data shape
    n_samples = 100 #samples of 1 window (1 window exists of X samples)
    resolution = 0.01 # time step between 2 samples
    SNR_dB = 50 # Signal to Noise ration in dB
    variation = 1 # 0 to 1 (0 to 100%), higher values possible
    #Gen = Datagenerator2.DataGenerator2(n_samples= n_samples,resolution = resolution,SNR_dB = SNR_dB, variation = variation, n_classes=20, useseed = False, seed = 5)
    Gen = Datagenerator2dontcare.DataGenerator2dontcare(n_samples= n_samples,resolution = resolution,SNR_dB = SNR_dB, variation = variation, n_classes=6, useseed = False, seed = 5)

    labels = Gen.class_names
    return labels

## calculate quartiles, means etc.    
def calc_upper_lower_band(matrix):
    #mean = matrix.mean(axis =0)
    median = np.percentile(matrix,50,axis = 0)
    lower = np.percentile(matrix,25,axis = 0)
    upper = np.percentile(matrix,75,axis = 0)
    lowlow = np.percentile(matrix,10,axis = 0)
    upup = np.percentile(matrix,90,axis = 0)
    return lowlow, lower, median, upper, upup

## load in Accuracy and confidence Data
def loadData(filepath,samples_trained_last=None,bagsize = 1):
    Data = np.load(filepath)
    Data = np.repeat(Data,bagsize,axis = 2)
    samples_trained = Data[0][0]
    if samples_trained_last is not None:
        if not np.array_equal(samples_trained,samples_trained_last):
            print('\x1b[6;30;42m'+ "ERROR: not the same amount of training Data"+'\x1b[0m')

    accuracy_scores = Data[1]
    av_cert = Data[2]
    error_cert = Data[3]
    true_cert = Data[4]
    accuracy_scores_mean = accuracy_scores.mean(0)
    av_cert_mean = np.mean(av_cert)
    error_cert_mean = error_cert.mean(0)
    true_cert_mean = true_cert.mean(0)
    return samples_trained,accuracy_scores,av_cert,error_cert,true_cert,accuracy_scores_mean,av_cert_mean,error_cert_mean,true_cert_mean

## load in Data from Time measurements
def loadTimes(filepath,samples_trained_last=None,bagsize=1):
    RawData = np.load(filepath)

    ## remove first run because it is slower (why ??) idk
    
    #Data = RawData[1:,:]
    Data = RawData


    ## 1st value = FST time 
    FST_time = Data[:,0]
    ## 2nd value = initial learning time + FST
    trainFST = Data[:,1]


    # repeat hier sinnvoll ? 
    # --> evtl. schon um verschiedene bag-Zeitmessungen zu vergleichen.
    Data = np.repeat(Data[:,2:],bagsize,axis = 1)#

    #add zero, FST and train time as first numbers

    #Times = np.empty([Data.shape[0],Data.shape[1]+3])
    Times = np.empty([Data.shape[0],Data.shape[1]])
    for i,run in enumerate(Data):   
        #Times[i] = np.append(0,np.append(FST_time[i],np.append(trainFST[i],run)))
        #Times[i] = np.append(0,run)
        #Times[i] = run - trainFST[i]
        Times[i] = run - run[0]
    return Times, FST_time, trainFST

## visualize Data with many boxplots \n
# If ax is None a new window is created \n
# Otherwise the results are plotted in addition in the existing one  
def boxplotfunc(ax, name, samples_trained, scores, av_cert, error_cert, true_cert, scores_mean, av_cert_mean, error_cert_mean, true_cert_mean):
    x = range(1,samples_trained.shape[0] + 1)
    color = ax.plot(x,scores_mean, label = "accuracy score with "+ str(name))[0].get_color()#,color = 'green') #str(learner.estimator))+str(learner.query_strategy)
    #ax.plot(samples_trained, scores_mean, label = "accuracy score")#str(learner.estimator))+str(learner.query_strategy)
    #ax.scatter(samples_trained, scores_mean)
    #ax.plot(samples_trained,true_cert_mean,label="True Pos Certainty")
    #ax.plot(samples_trained,error_cert_mean,label = "Error Certainty")
    #ax.scatter(x,scores_mean)
    #ax.plot(x,true_cert_mean,label="True Pos Certainty")
    #ax.plot(x,error_cert_mean,label = "Error Certainty")
    flierprops = dict(marker=',', markerfacecolor=color, markersize=4,linestyle='none')
    meanlineprops = dict(linestyle='solid', linewidth=2.5, color= color)
    medianprops = dict(linestyle='-', linewidth=0.1, color=color,)
    whiskerprops = dict(linewidth = 5, color = color,alpha = 0.01)
    boxprops = dict(color = color, alpha = 0.1)
    b = ax.boxplot(scores, showmeans = False,  meanline = False, showbox = True, showcaps = False, showfliers = False,flierprops = flierprops, meanprops = meanlineprops, medianprops = medianprops, whiskerprops = whiskerprops, boxprops= boxprops, patch_artist=True)
    for box in b['boxes']:
        box.set_facecolor(color)
    ax.plot(x,scores_mean,color = color)

## visualize Data with median and colored lines \n
# If ax is None a new window is created \n
# Otherwise the results are plotted in addition in the existing one  
def rangeplotfunc(ax, name, samples_trained, scores, av_cert=None, error_cert=None, true_cert=None, scores_mean=None, av_cert_mean=None, error_cert_mean=None, true_cert_mean=None):
    x = range(1,samples_trained.shape[0] + 1)
    lowlow, lower, mean, upper, upup = calc_upper_lower_band(scores)
    mins = scores.min(axis = 0)
    maxs = scores.max(axis=0)

    color = ax.plot(x,mean, label = " "+ str(name))[0].get_color()#,color = 'green')
    #ax.plot(x,lower,color = color,alpha = 0.3)
    #ax.plot(x,upper,color = color,alpha = 0.3)
    plt.fill_between(x,lower,upper,color = color, alpha = 0.4)
    #ax.plot(x,lowlow,color = color,alpha = 0.1)
    #ax.plot(x,upup,color = color,alpha = 0.1)
    #plt.fill_between(x,mins,maxs,color = color, alpha = 0.15)
    ax.plot(x,mins,color = color,alpha = 0.2)
    ax.plot(x,maxs,color = color,alpha = 0.2)

## load in Confusion Matrices 
def loadConfData(filepath,samples_trained_last=None):
    Confusionmatrices = np.load(filepath)
    samples_trained = np.sum(Confusionmatrices[0]) + Confusionmatrices.shape[0]
    if samples_trained_last[0] != None:
        if samples_trained.all() != samples_trained_last.all():
            print('\x1b[6;30;42m'+ "ERROR: not the same amount of training Data"+'\x1b[0m')
    return Confusionmatrices

## plot accuracy data over seen windows
def accuracyplot(Datafile,algoname,name,ax = None,bagsize = 1):
    if ax is None:
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)

    samples_trained = np.load(Datafile)[0][0]


    
    samples_trained, scores, av_cert, error_cert, true_cert, scores_mean, av_cert_mean, error_cert_mean, true_cert_mean = loadData(Datafile,samples_trained,bagsize)



    #plot
    #boxplotfunc(ax,"random sampling", samples_trained, scores, av_cert, error_cert, true_cert, scores_mean, av_cert_mean, error_cert_mean, true_cert_mean)
    rangeplotfunc(ax,name, samples_trained, scores, av_cert, error_cert, true_cert, scores_mean, av_cert_mean, error_cert_mean, true_cert_mean)




    ## 
        #randomPath = '/home/tob/git/al-on-ad/resultsAL/dontcare5classes10runs.npy'
        #certaintyPath = '/home/tob/git/al-on-ad/resultsAL/dontcare5classes10runs.npy'
        #certaintyPath = '/home/tob/git/al-on-ad/resultsAL/dontcare5classes100runs_certainty.npy'
        #randomPath = '/home/tob/git/al-on-ad/resultsAL/dontcare5classes100runs_random.npy'




    ## Settings
    x = range(1,samples_trained.shape[0] + 1)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 

    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   

    plt.ylim((-0.1,1.1))

    plt.yticks(fontsize=10)
    for y in range(0, 11, 2):    
        plt.plot(x, [y*0.1] * len(x), "--", lw=0.5, color="black", alpha=0.3) 
    plt.tick_params(axis="both", which="both",reset=True, bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)  

    distance = int(x[-1]/10)
    plt.xticks(range(0, x[-1], distance),[str(sample)  for sample in range(int(samples_trained[0]),int(samples_trained[-1])+1,distance)],fontsize = 10)

    plt.xlabel('number of windows')
    plt.ylabel('Accuracy / Certainty')
    ax.legend()
    #TODO: Testing
    plt.axvline(x=samples_trained[0], color="black", linewidth = 0.5, ymin =0, ymax= plt.ylim()[1])
    plt.figtext(0.01,0.04,"initial samples: "+ str(samples_trained[0]))
    ax.set_title('Performance of the classifier during the active learning: '+ algoname)
    return ax

def Timingplot(Timesfile,algoname,name,ax = None,bagsize = 1,n_initial = 100, n_total = 1000,humanAnnoTime = 0,derive = False):
    if ax is None:
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
    n_queries = int((n_total-n_initial)/bagsize)

    #samples_trained = np.arange(n_initial-2,n_queries*bagsize+n_initial+1,1)
    samples_trained = np.arange(n_initial,n_queries*bagsize+n_initial,1)

    #print(samples_trained.shape)
    Times, FST_time, trainFST = loadTimes(Timesfile,samples_trained,bagsize)

    print("FST_time:")
    print(FST_time.mean())
    print("")
    print("completeTime:")
    print(Times[:,-1].mean())
    print("")



    ## derivative
    if derive == True:
        #print(Times.shape)
        diffTimes = np.diff(Times)
        for i,time in enumerate(diffTimes):
            #print(time.shape)
            Times[i] = np.append(0,time)
            #print(time.shape)
        #print(Times.shape)


    #calculate human Annotation time
    AnnoTimes = np.empty(Times[0].shape) 
    for i,window in enumerate(samples_trained):
        AnnoTimes[i] = humanAnnoTime*(window-n_initial) #-n_initital to move the linear function to cut x-axis at n_initial
    AnnoTimes = np.maximum(AnnoTimes,0)    #cut off negative values
    for Timeline in Times:
        Timeline+= AnnoTimes

    rangeplotfunc(ax,name, samples_trained, Times)

    ## Settings
    x = range(1,samples_trained.shape[0] + 1)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 

    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   

    

    maxtime = Times.max()+0.1

    if plt.ylim()[1] < maxtime:
        plt.ylim((0,maxtime))

    plt.yticks(fontsize=10)
    #for y in range(0, int(maxtime*10), int(np.ceil(maxtime))):    
    #    plt.plot(x, [y*0.1] * len(x), "--", lw=0.5, color="black", alpha=0.3) 
    plt.tick_params(axis="both", which="both",reset=True, bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)  

    #upperend = plt.axvline(x=2,color="black", linewidth = 0.25,ymin =0, ymax= plt.ylim()[1]) # FST
    #plt.axvline(x=2+1,color="black", linewidth = 0.5,ymin =0, ymax= plt.ylim()[1]) #FST + initial learning

    #plt.figtext(0.14,0.1,"FST") #rechte untere Ecke
    #plt.figtext(0.155,upperend.get_ydata()[1]-0.01,"FST")

    distance = int(n_total/10)
    plt.xticks(range(2+1, n_total, distance),[str(sample)  for sample in range(int(n_initial),int(n_total)+1,distance)],fontsize = 10)

    #distance = int(x[-1]/10)
    #plt.xticks(range(0, x[-1], distance),[str(sample)  for sample in range(int(samples_trained[0]),int(samples_trained[-1])+1,distance)],fontsize = 10)


    plt.xlabel('number of windows')
    plt.ylabel('Time in Seconds')
    ax.legend()
    plt.figtext(0.01,0.04,"initial samples: "+ str(n_initial))
    
    ax.set_title('Performance of the classifier during the active learning: '+ algoname)
    plt.legend(loc='lower right')
    return ax

def Timestackedbarchart(Timesfile,algoname,name,ax = None,bagsize = 1,n_initial = 100, n_total = 1000,humanAnnoTime = 0):
    if ax is None:
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)

    n_queries = int((n_total-n_initial)/bagsize)
    samples_trained = np.arange(n_initial,n_queries*bagsize+n_initial,1)
    Times, FST_time, trainFST = loadTimes(Timesfile,samples_trained,bagsize)

    #calculate human Annotation time
    AnnoTimes = np.empty(Times[0].shape) 
    for i,window in enumerate(samples_trained):
        AnnoTimes[i] = humanAnnoTime*(window-n_initial) #-n_initital to move the linear function to cut x-axis at n_initial
    AnnoTimes = np.maximum(AnnoTimes,0)    #cut off negative values
    for Timeline in Times:
        Timeline+= AnnoTimes

    
    completeTimes = Times[:,-1] # only last value matters
    #completeTimes -= trainFST


    #print(Times.shape)
    #print(completeTimes)
    #print(completeTimes.shape)
    #print(trainFST)

    trainFST_mean = trainFST.mean()
    train_FST_std = trainFST.std()

    complete_mean = completeTimes.mean()
    complete_std = completeTimes.std()

    width = 0.35


    ax.bar(name,trainFST_mean,width,yerr = train_FST_std,label ='FST',color = 'red')
    ax.bar(name,complete_mean,width,yerr = complete_std, bottom = trainFST_mean ,label ='Rest', color = 'green')

    
    plt.ylabel('Time in Seconds')
    ax.legend(['FST', 'Rest'])
    return ax

def conf_rangeplotfunc(confusionmatrix,samples_trained,name):
    labels = getlabels()

    x = range(1,samples_trained.shape[0] + 1)
    lowlow, lower, mean, upper, upup = calc_upper_lower_band(confusionmatrix)
    mins = confusionmatrix.min(axis = 0)
    maxs = confusionmatrix.max(axis=0)
    n_classes = confusionmatrix.shape[1]
    #ax = plt.subplot(n_classes,1)
    #color = ax.plot(x,mean)[0].get_color()#,color = 'green')
    for i in range(n_classes):
        for j in range(n_classes):
            ax = plt.subplot(n_classes,n_classes,j+i*n_classes+1)
            color = ax.plot(x,mean[i,j],label = name)[0].get_color()#,color = 'green')
            #ax.plot(x,lower[i,j],color = color,alpha = 0.3)
            #ax.plot(x,upper[i,j],color = color,alpha = 0.3)
            plt.fill_between(x,lower[i,j],upper[i,j],color = color, alpha = 0.4)
            #ax.plot(x,lowlow[i,j],color = color,alpha = 0.1)
            #ax.plot(x,upup[i,j],color = color,alpha = 0.1)
            #plt.fill_between(x,lowlow[i,j],upup[i,j],color = color, alpha = 0.02)
            ax.plot(x,mins[i,j],color = color,alpha = 0.2)
            ax.plot(x,maxs[i,j],color = color,alpha = 0.2)
            #plt.fill_between(x,mins[i,j],maxs[i,j],color = color, alpha = 0.02)

            #Settings
            ax.spines["top"].set_visible(False)    
            #ax.spines["bottom"].set_visible(False)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(False) 
            ax.get_xaxis().tick_bottom()    
            ax.get_yaxis().tick_left()   

            
            
            limit = np.around(mean[i,:,-1].sum())
            #limit = np.around(maxs[i,j,:].max())
            #limit = np.around(upup[i,j,:].max())
            #limit = np.around(upper[i,j,:].max())

            _,top = plt.ylim()
            limit = max(limit,top-0.1)
            if limit < 1:
                limit = 1       
            plt.ylim((-0.01,limit+0.1))
            plt.yticks(fontsize=10)
            #for y in range(0, int(limit*10), int((limit*10)/5)):    
                #plt.plot(x, [y*0.1] * len(x), "--", lw=0.5, color="black", alpha=0.3) 
            if i != n_classes-1:
                plt.xticks([])
            else:
                plt.xlabel(labels[j])
            #plt.tick_params(axis="both", which="both",reset=True, bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)  
            
            if j == 0:
                h = plt.ylabel(labels[i],labelpad=25)
                h.set_rotation(0)

            #distance = int(x[-1]/10)
            #plt.xticks(range(0, x[-1], distance),[str(sample)  for sample in range(int(samples_trained[0]),int(samples_trained[-1])+1,distance)],fontsize = 10)   

def normalizeMatrix(matrix):
    for query in range(matrix.shape[-1]):
        for i in range(matrix.shape[0]):
            summe = matrix[:,i,query].sum()
            if summe == 0:
                matrix[:,i,query] = 0
            else: 
                matrix[:,i,query] = matrix[:,i,query]/summe
    #matrix = np.clip(matrix,0,1)
    return matrix

def normalizeData(matrix):
    for data in matrix:
        data = normalizeMatrix(data)
    return matrix

def confusionPlots(filename,algo,name,ax = None):
    # create figure
    if ax is None:
        ax = plt.figure(figsize=(12, 9))
    Datafile = os.path.join(base_path,filename)
    ConfDatafile = os.path.join(base_path,("Conf_"+filename))

    samples_trained = np.load(Datafile)[0][0]
    confusionmatrix = loadConfData(ConfDatafile,samples_trained)
    #confusionmatrix = normalizeData(confusionmatrix)
    conf_rangeplotfunc(confusionmatrix,samples_trained,name)


    #query = "certainty"
    #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    #Datafile = os.path.join(base_path,filename)
    #ConfDatafile = os.path.join(base_path,("Conf_"+filename))
    #samples_trained = np.load(Datafile)[0][0]
    #confusionmatrix = loadConfData(ConfDatafile,samples_trained)
    ##confusionmatrix = normalizeData(confusionmatrix)
    #conf_rangeplotfunc(confusionmatrix,samples_trained,query)



    plt.suptitle('Confusion Matrix: ' + algo)
    plt.figtext(0.01,0.04,"initial samples: "+ str(samples_trained[0]))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    return ax

def queryPerformance(base_path, filename,algo,name,ax = None, zip = False, init = None):
    if ax is None:
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
    queryOrderfile = os.path.join(base_path,("Queries_"+filename))
    poolFile = os.path.join(base_path,("Pools_"+filename))
    pools = np.load(poolFile)
    queries = np.load(queryOrderfile)
    

    #Normalize
    normqueries = np.empty([queries.shape[0],queries.shape[1],int(pools.max()+1)])
    for i,query in enumerate(queries): #Durchlauf i
        p_unique, p_counts = np.unique(pools[i], return_counts=True)
        for j,sample in enumerate(query): #question j
            sample = int(sample)
            q_unique, q_counts = np.unique(query[0:j+1], return_counts=True)
            if q_unique.shape[0] < p_unique.shape[0]: #not all classes seen yet
                tempVar = np.zeros(p_counts.shape)
                for l,k in enumerate(q_unique):
                    k = int(k)
                    tempVar[k] = q_counts[l]
                q_counts = tempVar
  
            for k in p_unique: #Class k
                k = int(k)
                normqueries[i,j,k] = q_counts[k]/p_counts[k]

    #plot
    samples_shown = queries[0]
    x = range(samples_shown.shape[0]+1)
    #print(samples_shown)
    if zip:  #print one curve, reprensentative for all 5 classes
        plotnormqueries = np.zeros(normqueries[:,:,0].shape)
        for i in range(int(pools.max())):
            plotnormqueries += normqueries[:,:,i]
        plotnormqueries = plotnormqueries/pools.max()
        #rangeplotfunc(ax,querystrat +" sampling", samples_shown, plotnormqueries)
        rangeplotfunc(ax,name, samples_shown, plotnormqueries)
    elif not zip:
        for i in range(int(pools.max()+1)):
            rangeplotfunc(ax,name+ getlabels()[i], samples_shown, normqueries[:,:,i])



    #rangeplotfunc(ax,querystrat +" sampling", samples_shown, normqueries[:,:,0])

    ## Settings
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False) 

    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   

    plt.ylim((-0.1,1.1))

    plt.yticks(fontsize=10)
    for y in range(0, 11, 2):    
        plt.plot(x, [y*0.1] * len(x), "--", lw=0.5, color="black", alpha=0.3) 
    plt.tick_params(axis="both", which="both",reset=True, bottom=False, top=False, labelbottom=True, left=False, right=False, labelleft=True)  

    distance = int(x[-1]/10)
    plt.xticks(range(0, x[-1], distance),[str(sample)  for sample in range(int(x[0]),int(x[-1])+1,distance)],fontsize = 10)

    plt.xlabel('number of windows')
    plt.ylabel('Proportion of seen Labels')
    ax.legend()
    
    

    #TODO: Testing
    if init is not None:
        plt.figtext(0.01,0.04,"initial samples: "+ str(init))#str(x[0]))
        plt.axvline(x=init, color="black", linewidth = 0.5, ymin =0, ymax= plt.ylim()[1])

    ax.set_title('Performance of the classifier during the active learning: '+ algo)

    return ax


################################
### Custom function collection
################################

def ROCKET_querycompare():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/GunPoint/")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/FaceAll/")
    n = 25
    bag = 1
    #algo = "Mine"
    algos = ["BOSS"]
    #algos = ["ROCKET"]

    #algos = ["BOSS", "ROCKET"]
    queries = ["random","certainty","entropy","uncertainty"]
    #queries = ["random","certainty"]
    #queries = ["certainty"]

    ## Accuracy
    ax = None
    for query in queries:
        #filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
        #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
        print(base_path)
        #Datafile = os.path.join(base_path,filename)
        #print(Datafile)
        #ax = accuracyplot(Datafile,algo,query,ax)

    ## Confusion Matrix
    ax= None
    for query in queries:
        pass
        #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
        #ax = confusionPlots(filename,algo,query,ax)

    ## Samples shown
    ax = None
    for algo in algos:
        for query in queries:
            filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
            #filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
            ax = queryPerformance(base_path,filename,algo,query+ " sampling",ax,zip= True,init = 100)
    plt.legend(loc='lower right')
    return ax

def BOSS_RF_KNN_compare():
    ax = None
    n = 25
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy")

    algo ="BOSS"
    query = "certainty"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"

    label = "BOSS certainty KNN"
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)

    base_path = os.path.join(os.path.dirname(__file__),"resultsAL")
    n = 10
    query = "certainty"
    filename = algo + "RF_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "BOSS certainty RF"
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)
    return ax

def ROCKET_normorNot():
    ax = None

    n = 25
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/ROCKET")
    #ax = None
    algo = "ROCKET"
    #query = "uncertainty"
    query = "certainty"
    filename = algo + "10k_kernels_RidgeNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "Ridge Normalized"
    ax = queryPerformance(filename,algo,label,ax,zip= True)
    filename = algo + "10k_kernels_RidgeNotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "Ridge Unnormalized"
    ax = queryPerformance(filename,algo,label,ax,zip= True)



    #filename = algo + "10k_kernels_RidgeNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    #ax = queryPerformance(filename,algo,query,ax,zip= True)
    #filename = algo + "10k_kernels_RidgeNotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    #query = "ROCKET Ridge certainty"
    #ax = queryPerformance(filename,algo,query,ax,zip= True)

    # --> NotNorm outperforms Norm


    ## Random Forest
    ### relativ langsam da nur auf einem Kern

    n =10
    query = "certainty"

    filename = algo + "10k_kernels_RF_Norm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "ROCKET RF Normalized"
    ax = queryPerformance(filename,algo,label,ax,zip= True)

    filename = algo + "10k_kernels_RF_NotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "ROCKET RF Not Normalized"
    ax = queryPerformance(filename,algo,label,ax,zip= True)



    ax.set_title("Performance of Normalized and Unnormalized ROCKET")
    return ax

def RandomForestTreeCompare():
    ax = None
    #unlimited
    query = "certainty"
    filename = algo + "10k_kernels_RF_NotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "unlimited"
    ax = queryPerformance(filename,algo,label,ax,zip= True)


    #100
    filename = algo + "10k_kernels_RF_limited100_NotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "Limit 100"
    ax = queryPerformance(filename,algo,label,ax,zip= True)

    #15
    filename = algo + "10k_kernels_RF_limited15_NotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "Limit 15"
    ax = queryPerformance(filename,algo,label,ax,zip= True)

    #10
    filename = algo + "10k_kernels_RF_limited10_NotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "Limit 10"
    ax = queryPerformance(filename,algo,label,ax,zip= True)

    #5
    filename = algo + "10k_kernels_RF_limited5_NotNorm_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "Limit 5"
    ax = queryPerformance(filename,algo,label,ax,zip= True)

    ax.set_title("Random Forest 100 Trees + ROCKET with limits in depth")
    #ax.legend(3*["unlimited"]+3*["100"]+3*["15"]+3*["10"]+3*["5"])
    #ax.legend(["unlimited"]+["100"]+["15"]+["10"])
    return ax

def BOSS_entropyqueryCompare():
    ax = None
    ax2 = None
    n = 25
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")

    algo ="BOSS"
    query = "certainty"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = "certainty sampling bagsize 1"
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)


    query = "imbEntropy"
    bagsizes = [1,5,10,15]

    for bag in bagsizes:
        filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
        label = "entropy sampling bagsize" + str(bag)
        ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)
        
    #for bag in bagsizes:
    #    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    #    Datafile = os.path.join(base_path,filename)
    #    ax2 = accuracyplot(Datafile,algo,query,ax2)

    return ax,ax2

def BOSS_bagsizeCompare():
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var1init10000total")

    n = 5
    algo = "BOSS"
    query = "certainty"
    bagsize = [1,5,10,15]
    ax = None
    for bag in bagsize:
        #filename = algo + "_dontcare5classes_bag" +str(bag) +str(n) +"runs_"+query+".npy"
        filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
        label = "bagsize: "+str(bag)
        ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)

    ## 20 dB
    bag = 15
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/20dB1var1init10000total")
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "20 dB Noise bagsize: "+str(bag)
    ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)
    
    return ax

def BOSS_bagsizeCompare_acc():
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var1init10000total")

    n = 5
    algo = "BOSS"
    query = "certainty"
    bagsize = [1,5,10,15]
    ax = None
    for bag in bagsize:
        #filename = algo + "_dontcare5classes_bag" +str(bag) +str(n) +"runs_"+query+".npy"
        filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
        Datafile = os.path.join(base_path,filename)
        label = "bagsize: "+str(bag)     
        ax = accuracyplot(Datafile,algo,label,ax,bag)
    return ax

def ROCKET_bagsizeCompare():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    n = 5
    algo = "ROCKET"
    query = "certainty"
    #bagsize = [1,5,10,15]
    bagsize = [1,5,10,15]
    ax = None
    for bag in bagsize:
        filename = algo + "Ridge_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
        label = "bagsize: "+str(bag)
        ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)

    return ax

def ROCKET_queryCompare():
    ax = ROCKET_bagsizeCompare()
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    n = 5
    algo = "ROCKET"
    query = "imbEntropy"
    bag = 15
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "imbalanced Entropy Sampling bag 15"
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)
    bag = 1
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "imbalanced Entropy Sampling bag 1"
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)
    return ax

def ROCKET_cert_entropy_mix():
    ax = ROCKET_queryCompare()
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    n = 1
    algo = "ROCKET"
    query = "imbEntropy2"
    bag = 10
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "imbalanced Entropy Sampling 2 bag 10"
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)
    return ax

def ROCKET_bagsizeCompare_acc():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    n = 5
    algo = "ROCKET"
    query = "certainty"
    bagsize = [1,5,10,15]
    ax = None
    for bag in bagsize:
        filename = algo + "Ridge_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
        Datafile = os.path.join(base_path,filename)
        label = "bagsize: "+str(bag)     
        ax = accuracyplot(Datafile,algo,label,ax,bag)
    return ax

def longROCKET(ax):
    n = 5
    algo = "ROCKET"
    query = "certainty"
    bag = 15

    ## 50 dB
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var1init10000total")
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "ROCKET 50 dB Noise bagsize 15"
    ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)

    n =1 
    ## 20 dB
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/20dB1var1init10000total")
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "ROCKET 20 dB Noise bagsize 15"
    ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)

    return ax

def NoisecompareBOSSROCKETbig():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var1init10000total")
    ax = None
    n = 5
    algo = "BOSS"
    query = "certainty"
    bag = 15

    ## BOSS 50dB
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "BOSS 50dB"
    ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)

    ## BOSS 20dB
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/20dB1var1init10000total")
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "BOSS 20dB"
    ax = queryPerformance(base_path,filename,algo,label,ax,zip= True)
    ax = longROCKET(ax)
    ax.set_title("Performance of the Classifiert during the active learning on a big Dataset: bagsize 15")

    return ax

def BOSS_certainty_Insight():
    ax = None
    n = 25
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/Gunpoint")

    algo ="BOSS"
    query = "certainty"
    Bagsize = 1
    #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(Bagsize)+"bag" +".npy"
    label = ""
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= False)
    ax.set_title(algo+" "+query+" sampling")
    plt.legend(loc='lower right')
    return ax

def ROCKET_certainty_Insight():
    ax = None
    n = 25
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")

    algo ="ROCKET"
    query = "certainty"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    label = ""
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= False)
    ax.set_title(algo+" "+query+" sampling")
    plt.legend(loc='lower right')
    return ax

def BOSS_entropy_bagsize_detail_compare():
    ax = None
    n = 25
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")

    algo ="BOSS"
    query = "imbEntropy"
    bag = 1
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "1 bag imbalanced Entropy sampling class "
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= False)

    bag = 15
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    label = "15 bag imbalanced Entropy sampling class "
    #ax = queryPerformance(base_path, filename,algo,label,ax,zip= False)

def BOSS_entropy_certainty_acc():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var1init10000total")

    ax = None
    n = 25
    algo = "BOSS"
    query = "certainty"
    bag = 1
    
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
    Datafile = os.path.join(base_path,filename)
    label = "certainty sampling"
    ax = accuracyplot(Datafile,algo,label,ax,5)

    query = "imbEntropy"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Datafile = os.path.join(base_path,filename)
    label = "entropy sampling"
    ax = accuracyplot(Datafile,algo,label,ax,bag)
    return ax

def BOSS_ROCKET_timing_compare():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")


    n = 10
    algo = "BOSS"
    query = "certainty"
    bag = 1
    ax = None

    
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo  
    ax = Timingplot(Timingfile,algo,label,ax,bag)


    algo ="ROCKET"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo  
    ax = Timingplot(Timingfile,algo,label,ax,bag)



    ax.set_title("Computation Time for a bagsize of "+str(bag) )
    return ax

def BOSS_bag_timing_compare():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")


    n = 10
    algo = "BOSS"
    #algo = "ROCKET"
    query = "certainty"
    bag = 1
    ax = None

    
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = "bagssize" + str(bag)
    ax = Timingplot(Timingfile,algo,label,ax,bag)


    bag = 10
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = "bagssize" + str(bag)
    ax = Timingplot(Timingfile,algo,label,ax,bag)




    ax.set_title("Computation Time for "+algo )
    return ax

def timing_overview_boss_rocket_bag():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init10000total")

    n = 10
    #n = 1
    algo = "BOSS"
    query = "certainty"
    bag = 1
    ax = None


    htime = 0
    derive = False


    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " bagssize" + str(bag)
    #ax = Timingplot(Timingfile,algo,label,ax,bag,humanAnnoTime=htime)


    
    bag = 10
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " bagssize" + str(bag)
    ax = Timingplot(Timingfile,algo,label,ax,bag,n_total = 1000,humanAnnoTime=htime,derive = derive)

    #return ax

    algo = "ROCKET"
    bag = 1

    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " bagssize" + str(bag)
    #ax = Timingplot(Timingfile,algo,label,ax,bag,humanAnnoTime=htime)


    bag = 10
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " bagssize" + str(bag)
    #ax = Timingplot(Timingfile,algo,label,ax,bag,humanAnnoTime=htime)

    ##Random Forest

    #n = 10
    #n = 1
    bag = 10
    filename = algo + "_RF" + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " RF"+ " bagssize" + str(bag) #+" (parallel)"
    ax = Timingplot(Timingfile,algo,label,ax,bag,n_total = 1000,humanAnnoTime=htime,derive = derive)


    #n = 5
    bag = 10
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " Ridge "+ "bagssize" + str(bag)
    ax = Timingplot(Timingfile,algo,label,ax,bag,n_total = 1000,humanAnnoTime=htime,derive = derive)




    #n = 6
    #n = 1
    bag = 10
    filename = algo + "_RF_parallel" + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " RF parallelized "+ " bagssize" + str(bag)
    #ax = Timingplot(Timingfile,algo,label,ax,bag,humanAnnoTime=htime,derive = derive)



    ax.set_title("Computation Time")
    return ax

def performance_boss_rocket_10bag():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    n = 25
    algo = "BOSS"
    query = "certainty"
    bag = 10
    ax = None


    #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    #filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +".npy"
    label = algo + " bagssize" + str(bag)
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True,init = 100)


    algo = "ROCKET"

    #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +".npy"
    label = algo + " RF (parallel) "+ " bagssize" + str(bag)
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)

    algo = "ROCKET"
    #ilename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    #filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +".npy"
    label = "ROCKET"+ " bagssize" + str(bag)
    ax = queryPerformance(base_path, filename,algo,label,ax,zip= True)


    ax.set_title("Computation Time")
    return ax


def longBOSS():
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var1init10000total")
    n = 6
    algo = "BOSS"
    query = "certainty"
    bag = 1
    ax = None

    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " bagssize" + str(bag)
    ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = 10000)

    return ax

def stackedBarCompare():
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")    #small
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init10000total")   #big


    #n = 10
    n = 1
    algo = "BOSS"
    query = "certainty"
    bag = 10
    ax = None
    htime = 0

    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " bagssize" + str(bag)
    ax = Timestackedbarchart(Timingfile,algo,label,ax,bag,humanAnnoTime=htime)


    algo = "ROCKET"
    filename = algo + "_RF" +"_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " RF (parallel) "+ " bagssize" + str(bag)
    ax = Timestackedbarchart(Timingfile,algo,label,ax,bag,humanAnnoTime=htime)


    filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
    Timingfile = os.path.join(base_path,("Times_"+filename))
    label = algo + " Ridge "+ " bagssize" + str(bag)
    ax = Timestackedbarchart(Timingfile,algo,label,ax,bag,humanAnnoTime=htime)


    return ax

def RealDatasetsPerformance():
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var5init100000total")
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var5init1000total")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/GunPoint/")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/FaceAll/")
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/Crop/")

    n = 25
    #n = 1
    bag = 10
    #algo = "Mine"
    #algos =  ["BOSS"]
    #algos = ["ROCKET"]

    algos = ["BOSS", "ROCKET"]
    #queries = ["random","certainty","entropy","uncertainty"]
    #queries = ["random","certainty"]
    queries = ["certainty"]

    ## Samples shown
    ax = None
    for algo in algos:
        for query in queries:
            #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
            filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
            label = algo + " " + query + " bagsize " + str(bag)
            ax = queryPerformance(base_path,filename,algo,label,ax,zip= True,init = 5)
        #n = 1
    return ax

def RealDatasetsTiming():
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total") 
    #n_windows = 1000
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var5init50000total") 
    #n_windows = 50000
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/GunPoint/") 
    #n_windows = 50
    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/FaceAll/") 
    n_windows = 560
    #base_path = os.path.join(os.path.dirname(__file__),"resultsAL/Crop/")
    #n_windows = 7200 

    n = 1
    #n = 5
    #n = 11
    bag = 100
    #algo = "Mine"
    #algos = ["BOSS"]
    #algos = ["ROCKET"]

    algos = ["BOSS", "ROCKET"]
    #queries = ["random","certainty","entropy","uncertainty"]
    #queries = ["random","certainty"]
    queries = ["certainty"]

    ## Timing
    htime = 0
    derive = False
    ax = None
    for algo in algos:
        for query in queries:
            #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
            filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
            Timingfile = os.path.join(base_path,("Times_"+filename))
            print(filename)
            label = algo + " " + query
            #ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = 10000)
            ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = n_windows, humanAnnoTime=htime, derive = derive)
        n = 1
    return ax

def BigBOSSTimingCompare():

    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var5init10000total") 
    n_windows = 10000
    n = 5
    bag = 10
    algos = ["ROCKET","BOSS"]
    #algos = ["BOSS","ROCKET"]
    queries = ["certainty"]


    ##Timing
    htime = 0
    derive = True
    ax = None
    for algo in algos:
        for query in queries:
            #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
            filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
            Timingfile = os.path.join(base_path,("Times_"+filename))
            print(filename)
            label = algo + " " + query
            #ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = 10000)
            #ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = n_windows, humanAnnoTime=htime, derive = derive)

    base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var5init100000total") 
    n_windows = 100000
    n = 1
    bag = 100


    for algo in algos:
        for query in queries:
            #filename = algo + "_dontcare5classes_" +str(n) +"runs_"+query+".npy"
            filename = algo + str(n) +"runs_"+query +"_"+str(bag)+"bag" +".npy"
            Timingfile = os.path.join(base_path,("Times_"+filename))
            print(filename)
            label = algo + " " + query
            #ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = 10000)
            ax = Timingplot(Timingfile,algo,label,ax,bag,n_initial = 5, n_total = n_windows, humanAnnoTime=htime, derive = derive)
    return ax


#base_path = os.path.join(os.path.dirname(__file__),"resultsAL/50dB1var100init1000total")
base_path = os.path.join(os.path.dirname(__file__),"resultsAL")
imgfile = os.path.join(base_path,"AL_result.svg")
#imgfile = os.path.join(base_path,"AL_result.eps")
#imgfile = os.path.join(base_path,"AL_result.pdf")
tikzfile = os.path.join(base_path,"AL_result.tex")



performance_boss_rocket_10bag()



#plt.savefig(imgfile, format='eps')
#tikzplotlib.clean_figure()
#tikzplotlib.save(tikzfile)

plt.savefig(imgfile)
plt.show()


