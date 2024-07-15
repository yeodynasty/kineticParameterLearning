# Methods added to original file from: https://github.com/JBEI/KineticLearning
import pandas as pd
from IPython.display import display
import numpy as np
from tpot import TPOTRegressor
from scipy.interpolate import interp1d
from scipy.integrate import odeint,ode
import matplotlib.pyplot as plt
import seaborn as sns

# function to add noise to data
def addNoise(data,noise, noiseSeed):
    '''
    data: List with each element containing a list of values
    noise: as a fraction of datapoint value
    noiseSeed: any numerical input as seed for random number generation
    returns values with added noise 
         drawn from Normal distribution with mean value = 0 
         & standard deviation (scale) equal to an input fraction (noise) of datapoint value  
    '''
    
    # made generated values replicable
    np.random.seed(noiseSeed)

    return [[value + np.random.normal(scale=abs(noise*value)) for value in line] for line in data]

def addNoise2(data, noise, noiseSeed):
    '''
    Variation of addNoise function above 
          catering to a data list with element containing a number
    '''
    Z=[]
    Z.append(data)
    Z = addNoise(Z, noise, noiseSeed)
    noisyData = Z[0]
    
    return noisyData

def readARTrecommendationFile ( currentDesign_Dict, enzymeOrder_List, ARTrecommendation_csvFile):
    
    #--- function merge designs from ART & ad-hoc designs using SAME enzyme naming
    # WARNING: ad-hoc strains will be OVER-WRITTEN by strains in ART recommendation with same naming 
    # enzymeOrder_List: order of enzyme in currentDesign_Dict
    # ART file: any enzyme order
    
    inFile=open(ARTrecommendation_csvFile,"r")
    line=inFile.readline() # skip comment line
    
    while inFile:
        
        line=inFile.readline()
        
        if (len(line)> 0 ): # for each ART recommended strain
            
            s=line.split(',')
            strain=s[5] ## strain column in file
            summaryCol=s[8] ## summary column in file
            #print (strain + ': ' + summaryCol)
            
            sc=summaryCol.split('_')
            
            # get enzyme design
            enzymeDesign= {}
            for i in range(0,len(sc),2): # kl: every even number index
                enzyme=sc[i]
                design=sc[i+1]
                enzymeDesign[enzyme]=int(design)
            
            #get design in same order as enzymeOrder_List
            designList=[]
            for enzyme in enzymeOrder_List:
                designList.append(enzymeDesign[enzyme])
            
            # OVERWRITE any same strain in currentDesign
            currentDesign_Dict[strain]=designList
            print ('ART design for ' + strain + ' read in:')
            print (designList)
            print ('\n')

            
        else:
            print ("Completed reading of")
            print (ARTrecommendation_csvFile +'\n-------------------------------------------\n')
            break
    inFile.close()
    return currentDesign_Dict


# get enzyme conc at time t
def leaky_hill_fcn(t,kf,km,kl):
        return kf*t/(km + t) + kl

# get time series conc for an enzyme
def getEnzymeTimeSeries (timePoints,kf,km,kl):
    e = []
    for t in timePoints:
        e.append(leaky_hill_fcn(t,kf,km,kl))
    return e

# get time series conc for various enzymes with ke:[kf,km,kl] appended together
def getAllEnzymesTimeSeries (timePoints, ke):
    e=[]
    for i in range(int(len(ke)/3)): # iter thru 9 enzymes with individual [kf,km,kl]
        #get [kf,km,kl] 
        #HC: k[0-2] for i=0, k[3-5] for i=2,...
        kfkmkl = ke[3*i:3*(i+1)]
        e.append(getEnzymeTimeSeries(timePoints,*kfkmkl))
    return e

# get vector of enzyme conc at time t
def proteomicsData(t,ke): #HC: takes in ke values
    e = []
    for i in range(int(len(ke)/3)): # iter thru 9 enzymes with individual [kf,km,kl]
        #HC: k[0-2] for i=0, k[3-5] for i=2,...
        gains = ke[3*i:3*(i+1)] #HC: [kf,km,kl]        
        # *gains takes indexed elements as separate variables
        e.append(leaky_hill_fcn(t,*gains))        
    return e 

def plotAllEnzymesTimeSeries(controls, listEnzymeTimeSeries,timePoints,pdf=False):
    from matplotlib import pyplot as plt
    for i in range(len(listEnzymeTimeSeries)): # iter thru enzymes
        print(controls[i])
        plt.plot(timePoints,listEnzymeTimeSeries[i],color='blue')
        if pdf==True:
            plt.gca().set_facecolor('none') # transparent background
            plt.title(controls[i], fontsize=18)
            #plt.xlabel('', fontsize=24)
            #plt.ylabel('', fontsize=24)
            plt.savefig('figures/'+controls[i] + '.png', transparent=True)
        plt.show()
    plt.close()

def plotSimulatedKinetics(all_metabolites, sol,timePoints):
    '''
    For exploring metabolite dynamics from the solution of ODE simulation
    
    all_metabolites: list of strings for labelling individual plots
    
    timePoints: list of time
    
    sol: list of temporal elements,
             with each element containing a list of metabolite conc
             at the corresponding time
    '''
    listMetTimeSeries =[]        
    for j in range( len(all_metabolites) ):
        metaboliteTimeSeries=[]
        for k in range(len (timePoints) ):
            val = sol[k][j]
            metaboliteTimeSeries.append(val)
        listMetTimeSeries.append(metaboliteTimeSeries)

    plotAllEnzymesTimeSeries(all_metabolites, listMetTimeSeries,timePoints,pdf=True)

def smoothenTimeSeries2(dataType,df_raw,strainList,numberDataPtsNeeded,cubicSpline=True):
    #dataType: 'states' or 'controls'
    #--------------------------------------------------------------Prepare dataframe with new time points
    dfa=df_raw.copy() #make copy for alteration
    
    #get ALL possible time points
    times = dfa.index.get_level_values(1)
    
    #define the required number of evenly-spaced time points betw max & min time
    new_times = np.linspace(min(times),max(times),numberDataPtsNeeded)
    
    #Build New Indecies using new time points
    strain_name = set(dfa.index.get_level_values(0))
    new_indecies = pd.MultiIndex.from_product([strain_name,new_times])
    
    #Reindex based on new indices
    dfa = dfa.reindex(dfa.index.union(new_indecies))
    dfa.index.names = ['Strain','Time']
    #dfa = dfa.interpolate() #HC: Fill NAN values, linear method by default & only method supported for MultiIndexes
    
    #Remove indices & corresponding column values based on 'times'
    #dfa.index = dfa.index.droplevel(0) # '0' is level for 'Strain' that is being removed
    times_to_remove = set(times) - (set(times) & set(new_times)) # MAY BE EMPTY if times is subset of new_times
    dfa = dfa.loc[~dfa.index.isin(times_to_remove)] # index.isin return boolean array where the index values are found in times_to_remove
    #display (dfa)

    #-------------------------------------------------------------Smoothening & new data into new dataframe
    xs=new_times.tolist() # new time list
    
    dfa=smoothen(xs,df_raw, dfa, strainList, dataType,cubicSpline=cubicSpline)
    #display (dfa)
    
    return dfa




def read_metaboliteRawData(csv_path,states,time='Time',strain='Strain'):
    '''Put DataFrame into the TSDF format. #HC: time series dataframe? abbr unknown to python
    
    The input csv or dataframe should have a 
    column for time and every state and control
    variable input for that time. Optional Columns are
    "Replicate" and "Strain".
    
    '''
    
    #Load Raw Data
    df = pd.read_csv(csv_path)
    #df.head() #HC: check content of dataframe
	#df.info() #HC: get column data format & size
	#df.describe() #HC: column data statistics,e.g. mean, 25%, etc

    #Keep only data of selected Columns
    df = df[[strain,time] + states]    

    #Set Time Column to Float
	#HC: seems like already so without further action
    
    #HC: Declare columns
    df.columns = ['Strain','Time'] + states #HC: appear to be superceded by df.columns below
	#df.head() #HC: check content of dataframe
    
    #HC: define new index based on strain + time
    df = df.set_index(['Strain','Time'])
    
	#HC: create list of tuples (tuple: immutable item consisting of objects separated by ',' ('A','B')
    columns = [('states',state) for state in states]
    # Output:
	# [('states', 'Acetyl-CoA'),
    # ('states', 'HMG-CoA'),...
    
    df.columns = pd.MultiIndex.from_tuples(columns) #HC: columns of df relabelled by multi-index
    
    return df



def read_enzymeRawData(csv_path,controls,time='Time',strain='Strain'):
    '''Put DataFrame into the TSDF format. #HC: time series dataframe? abbr unknown to python
    
    The input csv or dataframe should have a 
    column for time and every state and control
    variable input for that time. Optional Columns are
    "Replicate" and "Strain".
    
    '''
    
    #Load Raw Data
    df = pd.read_csv(csv_path)
    #df.head() #HC: check content of dataframe
	#df.info() #HC: get column data format & size
	#df.describe() #HC: column data statistics,e.g. mean, 25%, etc

    #Keep only data of selected Columns
    df = df[[strain,time] +controls]    

    #Set Time Column to Float
	#HC: seems like already so without further action
    
    #HC: Declare columns
    df.columns = ['Strain','Time'] + controls #HC: appear to be superceded by df.columns below
	#df.head() #HC: check content of dataframe
    
    #HC: define new index based on strain + time
    df = df.set_index(['Strain','Time'])
    
	#HC: create list of tuples (tuple: immutable item consisting of objects separated by ',' ('A','B')
    columns = [('controls',control) for control in controls]
    # Output:
    # ('controls', 'AtoB'),
    # ('controls', 'GPPS'),...]
    
    df.columns = pd.MultiIndex.from_tuples(columns) #HC: columns of df relabelled by multi-index
    
    return df


def smoothen(xs, df_raw, dfa, strainList, dataType,cubicSpline=True):
    
    # dataType: either 'states' or 'controls'
    
    from scipy.interpolate import CubicSpline
    from scipy.interpolate import pchip
    from math import isnan
    
    for strain in strainList: #iter thru list of strain str
        
        for biomolecule in df_raw[dataType].columns: # iter thru list of biomolecule str
            
            # create ordered y and time lists
            y=df_raw[dataType,biomolecule].loc[strain] # define y
            yval=y.values.tolist() # convert to y value list
            timeSeries=df_raw[dataType,biomolecule].loc[strain] #define time (NOT dictionary)
            orderedTimeList=timeSeries.keys().tolist()#convert to time list
            
            # using dictionary comprehension to convert lists to dictionary
            timeSeriesDict = {orderedTimeList[i]: yval[i] for i in range(len(orderedTimeList))}

            # remove data pairs with y=NaN values
            clean_timeSeriesDict = {k: timeSeriesDict[k] for k in timeSeriesDict if not isnan(timeSeriesDict[k])}
            xclean=list(clean_timeSeriesDict.keys()) # prepare data pairs in ordered list format (ONLY IN PY 3.6 & ABOVE)
            yclean=list(clean_timeSeriesDict.values()) # prepare data pairs in ordered list format (ONLY IN PY 3.6 & ABOVE)
            
            if cubicSpline: # OPTION 1: Generate MONOTONIC cubic spline fit if data, i.e., yclean, is spare
                
                #cs=CubicSpline(xclean, yclean) # generate spline function based on LOESS interpolation
                cs=pchip(xclean, yclean) 
                ys=cs(xs)# interpolate y based on spline function of new set of time
            
            else: # OPTION 2: Generate best loess fit based on leave-one-out CV
                
                ys, degBest, nPointsBest, pcRMSE=bestLoess(xclean,yclean,xs)
                #display (ys)
                #cs=CubicSpline(xs, yLoess) # generate spline function based on LOESS interpolation
                #ys=cs(xs)# interpolate y based on spline function of new set of time
            
            # update dfa with smoothened data
            dfa[dataType,biomolecule].loc[strain]=ys
    return dfa


def smoothenTimeSeries(df_raw,strainList,numberDataPtsNeeded,cubicSpline=True):
    
    #--------------------------------------------------------------Prepare dataframe with new time points
    dfa=df_raw.copy() #make copy for alteration
    
    #get ALL possible time points
    times = dfa.index.get_level_values(1)
    
    #define the required number of evenly-spaced time points betw max & min time
    new_times = np.linspace(min(times),max(times),numberDataPtsNeeded)
    
    #Build New Indecies using new time points
    strain_name = set(dfa.index.get_level_values(0))
    new_indecies = pd.MultiIndex.from_product([strain_name,new_times])
    
    #Reindex based on new indices
    dfa = dfa.reindex(dfa.index.union(new_indecies))
    dfa.index.names = ['Strain','Time']
    #dfa = dfa.interpolate() #HC: Fill NAN values, linear method by default & only method supported for MultiIndexes
    
    #Remove indices & corresponding column values based on 'times'
    #dfa.index = dfa.index.droplevel(0) # '0' is level for 'Strain' that is being removed
    times_to_remove = set(times) - (set(times) & set(new_times)) # MAY BE EMPTY if times is subset of new_times
    dfa = dfa.loc[~dfa.index.isin(times_to_remove)] # index.isin return boolean array where the index values are found in times_to_remove
    #display (dfa)

    #-------------------------------------------------------------Smoothening & new data into new dataframe
    xs=new_times.tolist() # new time list
    
    dfa=smoothen(xs,df_raw, dfa, strainList, 'states',cubicSpline=cubicSpline)
    dfa=smoothen(xs,df_raw, dfa, strainList, 'controls',cubicSpline=cubicSpline)
    #display (dfa)
    
    return dfa


def bestLoess(xclean,yclean,xs):
    
    # xclean & yclean must NOT have NaN elements
    
    from loess.loess_1d import loess_1d
    from math import sqrt
    from math import isnan
    from sklearn.metrics import mean_squared_error
    from statistics import mean
    
    #col5: initial LOESS parameters & score
    RMSEbest=float('inf')
    degBest=-1
    nPointsBest=-1
    for deg in [1,2]: # iter degree of local polynomial fitting 
        
        for k in range(21):#len(xclean)): # iter number of data points to use in local fitting
            #col 13
            yPredicted=[]
            yActual=[]
            nPoints=k+1
            for j in range(len(xclean)): # loop to generate x & y data subsets with jth element removed
                
                # Prepare data x & y subset
                xCV=xclean.copy() # duplicate data
                xCV.pop(j) # remove jth element
                xCVarray=np.array(xCV) # list to array format for loess_1d
                
                yCV=yclean.copy() # same for y
                yCV.pop(j)
                yCVarray=np.array(yCV)
                
                 # prepare x[j] in array format
                xcleanj=[] 
                xcleanj.append(xclean[j])
                xcleanjArray=np.array(xcleanj)
                
                #LOESS prediction of y[j] based on x & y data subsets
                xTrial, yTrial, wTrial = loess_1d(xCVarray, yCVarray, xnew=xcleanjArray, degree=deg, npoints=nPoints)
                
                #Store yPredicted & yActual for jth element
                if not isnan(yTrial):
                    yPredicted.extend(yTrial.tolist())
                    yActual.append(yclean[j])
                    #ycleanj=[]  # prepare y data to be evaluated
                    #ycleanj.append(yclean[j])
                    #ycleanjArray=np.array(ycleanj)
                    #yActual.append(ycleanjArray.tolist())

            #col13: Evaluate RMSE for parameters combination
            RMSE = sqrt(mean_squared_error(yActual, yPredicted))
            
            # Store LOESS score & parameters, if RMSE < BEST
            if (RMSE < RMSEbest):
                RMSEbest=RMSE
                degBest=deg
                nPointsBest=nPoints
    

    #col5 Compute pcRMSE
    yMean=mean(yclean)
    pcRMSE= (RMSEbest/yMean)*100
    print('degBest:'+str(degBest)+', nPointsBest:'+str(nPointsBest)+', pcRMSE:'+str(pcRMSE))
    
    # LOESS fitting based on degBest & nPointsBest
    xcleanArray=np.array(xclean) # loess_1d only accepts array NOT list
    ycleanArray=np.array(yclean)
    xsArray=np.array(xs)
    
    if degBest != -1:
        xs, ys, wout = loess_1d(xcleanArray, ycleanArray, xnew=xsArray, degree=degBest, npoints=nPointsBest)
    else: # No best model
        xs, ys, wout = loess_1d(xcleanArray, ycleanArray, xnew=xsArray, degree=1, npoints=len(xclean))
    
    return ys, degBest, nPointsBest, pcRMSE


def augment_data(tsdf,strainList,n=200,plotEffect=True):
    '''Augment the time series data for improved fitting.
    
    The time series data points are interpolated to create
    smooth curves for each time series and fill in blank 
    values.
    '''
    
    def augment(df):
              
        #HC: get time points
        times = df.index.get_level_values(1)
        #HC: define n number of evenly-spaced time points betw max & min time
        new_times = np.linspace(min(times),max(times),n)
        
        #Build New Indecies using new time points
        strain_name = set(df.index.get_level_values(0))
        new_indecies = pd.MultiIndex.from_product([strain_name,new_times])
        
        #Reindex the Data Frame & Interpolate New Values
        df = df.reindex(df.index.union(new_indecies))
        df.index.names = ['Strain','Time']
        df = df.interpolate() #HC: Fill NAN values, linear method by default & only method supported for MultiIndexes
        
        #Remove Old Indecies
        df.index = df.index.droplevel(0)
        times_to_remove = set(times) - (set(times) & set(new_times))
        df = df.loc[~df.index.isin(times_to_remove)]
        #display (df)
        return df
    
    dfo=tsdf.copy()          
    tsdf = tsdf.groupby('Strain').apply(augment)
    
    #smoothed_measurement = savgol_filter(interpolated_measurement,7,2)
    
    #HC: Comparison plot after augmentation
    #display(dfo)
    #display(tsdf)
    if plotEffect:
        compareTimeSeriesAfterPreprocess(dfo,tsdf,'states',strainList,preprocessLabel='Ater augmentation')
        
    return tsdf


def preprocess_data2(df,strainList,impute=True,augment=None,est_derivative=True,smooth=False,n=None):
    
    #Sample num_strains Without Replacement
    if n is not None:
        strains = np.random.choice(df.reset_index()['Strain'].unique(),size=n)
        df = df.loc[df.index.get_level_values(0).isin(strains)]
    
    #Impute NaN Values using Interpolation
    if impute:
        df = df.groupby('Strain').apply(lambda group: group.interpolate())
    
    #Augment the data using an interpolation scheme
    if augment is not None:
        df = augment_data(df,strainList,n=augment,plotEffect=True)
    
    #Estimate the Derivative
    if est_derivative:
        df = estimate_state_derivative(df)
    
    return df


def NaN_table(df):
    """
    gives dataframe to preview missing values & % of missing values in each column
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns


def read_rawData(csv_path,states,controls,time='Time',strain='Strain'):
    '''Put DataFrame into the TSDF format. #HC: time series dataframe? abbr unknown to python
    
    The input csv or dataframe should have a 
    column for time and every state and control
    variable input for that time. Optional Columns are
    "Replicate" and "Strain".
    
    '''
    
    #Load Raw Data
    df = pd.read_csv(csv_path)
    #df.head() #HC: check content of dataframe
	#df.info() #HC: get column data format & size
	#df.describe() #HC: column data statistics,e.g. mean, 25%, etc

    #Keep only data of selected Columns
    df = df[[strain,time] + states+controls]    

    #Set Time Column to Float
	#HC: seems like already so without further action
    
    #HC: Declare columns
    df.columns = ['Strain','Time'] + states + controls #HC: appear to be superceded by df.columns below
	#df.head() #HC: check content of dataframe
    
    #HC: define new index based on strain + time
    df = df.set_index(['Strain','Time'])
    
	#HC: create list of tuples (tuple: immutable item consisting of objects separated by ',' ('A','B')
    columns = [('states',state) for state in states] + [('controls',control) for control in controls]
    # Output:
	# [('states', 'Acetyl-CoA'),
    # ('states', 'HMG-CoA'),...
    # ('controls', 'AtoB'),
    # ('controls', 'GPPS'),...]
    
    df.columns = pd.MultiIndex.from_tuples(columns) #HC: columns of df relabelled by multi-index
    
    return df



def plotRawTimeSeries(dfo,strainList):
    #HC: Compare time series data ('states' columns) b4 & after preprocessing
	#for strain in list 
    #e.g. after data augmentation

    #import matplotlib.pyplot as plt
    #plt.style.use('seaborn-whitegrid')
    from matplotlib.pyplot import cm
    import numpy as np
    for metabolite in dfo['states'].columns: #HC: 'states': list of metabolite strings
        
        plt.figure()
        ax = plt.gca()
        
        #Metabolite name
        name1 = ''.join([char for char in metabolite if char != '/']) #HC: remove potential '/' from METABOLITE str
        print (name1)
        
        #prepare datapoint colors
        colors = iter(cm.rainbow(np.linspace(0, 1, len(strainList))))
        
        for strain in strainList: #HC: list of strain strings
            
            c = next(colors) # change color
            
            name2 = ''.join([char for char in strain if char != '/']) #HC: remove potential '/' from STRAIN str
            #dfo['states'].loc[strain].reset_index().plot(x='Time',y=metabolite, ax=ax,label=name2)
            dfo['states'].loc[strain].reset_index().plot(kind='scatter', x='Time',y=metabolite,color=c, ax=ax,label=name2)
        plt.show()
        plt.close()


def compareTimeSeriesAfterPreprocess(dfo, dfa,dataType,strainList,preprocessLabel='After preprocessing'):
    #HC: Compare time series data ('states' columns) b4 & after preprocessing
	#for strain in list 
    #e.g. after data augmentation

    #import matplotlib.pyplot as plt
    #plt.style.use('seaborn-whitegrid')
    for strain in strainList: #HC: list of strain strings
        name2 = ''.join([char for char in strain if char != '/']) #HC: remove potential '/' from STRAIN str
        print ('Strain ' + name2)   
        for metabolite in dfo[dataType].columns: #HC: dataType: 'states' or 'controls'
            plt.figure()
            ax = plt.gca()
            dfo[dataType].loc[strain].reset_index().plot(kind='scatter', x='Time',y=metabolite, color='red', s=16, ax=ax,label='Original data')
            dfa[dataType].loc[strain].reset_index().plot(kind='scatter', x='Time',y=metabolite, color='black', s=2, ax=ax,label=preprocessLabel)
            #dfo[dataType].loc[strain].reset_index().plot(kind='scatter', 'Time',metabolite, 'ro',label='Original data')
            #dfa[dataType].loc[strain].reset_index().plot(kind='scatter', 'Time',metabolite, 'g-',label=preprocessLabel)
            
            name1 = ''.join([char for char in metabolite if char != '/']) #HC: remove potential '/' from str
            name=name1+'_'+name2+'_'
            
            plt.savefig('figures/' + name + preprocessLabel + '.pdf',transparent=False) #HC: save as pdf
            plt.show()
            plt.close()

#Decorators
def evenly_space(fun,times):
    '''Decorate Functions that require even spacing.'''
    pass


def estimate_state_derivative(tsdf,window,polynomial):
    '''Estimate the Derivative of the State Variables'''
    
    #Check if a vector is evenly spaced
    evenly_spaced = lambda x: max(set(np.diff(x))) - min(set(np.diff(x))) < 10**-5
    
    #Find the difference between elements of evenly spaced vectors
    delta = lambda x: np.diff(x)[0]

    from scipy.signal import savgol_filter

    def estimate_derivative(tsdf):
        state_df = tsdf['states']
        times = state_df.index.get_level_values(1)
        diff = delta(times)
        
        #Find Derivative of evenly spaced data using the savgol filter
        savgol = lambda x: savgol_filter(x,window,polynomial,deriv=1,delta=diff)
        
        if evenly_spaced(times):
            state_df = state_df.apply(savgol)      
        else:     
            state_df = state_df.apply(savgol_uneven)
            
        #Add Multicolumn
        state_df.columns = pd.MultiIndex.from_product([['derivatives'],state_df.columns])

        #Merge Derivatives Back
        tsdf = pd.merge(tsdf, state_df,left_index=True, right_index=True,how='left')

        return tsdf
    
        
    tsdf = tsdf.groupby('Strain').apply(estimate_derivative)
    return tsdf

def estimate_state_derivative52(tsdf):
    '''Estimate the Derivative of the State Variables'''
    
    #Check if a vector is evenly spaced
    evenly_spaced = lambda x: max(set(np.diff(x))) - min(set(np.diff(x))) < 10**-5
    
    #Find the difference between elements of evenly spaced vectors
    delta = lambda x: np.diff(x)[0]

    from scipy.signal import savgol_filter

    def estimate_derivative(tsdf):
        state_df = tsdf['states']
        times = state_df.index.get_level_values(1)
        diff = delta(times)
        
        #Find Derivative of evenly spaced data using the savgol filter
        savgol = lambda x: savgol_filter(x,5,2,deriv=1,delta=diff)
        
        if evenly_spaced(times):
            state_df = state_df.apply(savgol)      
        else:     
            state_df = state_df.apply(savgol_uneven)
            
        #Add Multicolumn
        state_df.columns = pd.MultiIndex.from_product([['derivatives'],state_df.columns])

        #Merge Derivatives Back
        tsdf = pd.merge(tsdf, state_df,left_index=True, right_index=True,how='left')

        return tsdf
    
        
    tsdf = tsdf.groupby('Strain').apply(estimate_derivative)
    return tsdf

def estimate_state_derivative73(tsdf):
    '''Estimate the Derivative of the State Variables'''
    
    #Check if a vector is evenly spaced
    evenly_spaced = lambda x: max(set(np.diff(x))) - min(set(np.diff(x))) < 10**-5
    
    #Find the difference between elements of evenly spaced vectors
    delta = lambda x: np.diff(x)[0]

    from scipy.signal import savgol_filter

    def estimate_derivative(tsdf):
        state_df = tsdf['states']
        times = state_df.index.get_level_values(1)
        diff = delta(times)
        
        #Find Derivative of evenly spaced data using the savgol filter
        savgol = lambda x: savgol_filter(x,7,3,deriv=1,delta=diff)
        
        if evenly_spaced(times):
            state_df = state_df.apply(savgol)      
        else:     
            state_df = state_df.apply(savgol_uneven)
            
        #Add Multicolumn
        state_df.columns = pd.MultiIndex.from_product([['derivatives'],state_df.columns])

        #Merge Derivatives Back
        tsdf = pd.merge(tsdf, state_df,left_index=True, right_index=True,how='left')

        return tsdf
    
        
    tsdf = tsdf.groupby('Strain').apply(estimate_derivative)
    return tsdf

#Reconstruct the curve using the derivative (Check that derivative Estimates are Close...)
def check_derivative(tsdf):
    '''Check the Derivative Estimates to Make sure they are good.'''
    
    #First Integrate The Derivative of Each Curve Starting at the initial condition
    
    for name,strain_df in tsdf.groupby('Strain'):
        #display(strain_df['derivatives'].tail)
        
        #for pdf naming
        s=strain_df.index.get_level_values('Strain').unique().tolist() #THE strain in list
        strain=''.join(s) #THE strain in str
        
        display ('Time profile from derivative vs. actual profile for '+strain)
        
        times = strain_df.index.get_level_values(1)
        dx_df = strain_df['derivatives'].apply(lambda y: interp1d(times,y,fill_value='extrapolate'))
        dx = lambda y,t: dx_df.apply(lambda x: x(t)).values
        x0 = strain_df['states'].iloc[0].values
        
        #Solve Differential Equation
        result = odeint(dx,x0,times)
        trajectory_df = pd.DataFrame(result,columns=strain_df['states'].columns)
        trajectory_df['Time'] = times
        trajectory_df = trajectory_df.set_index('Time')
        
        for column in strain_df['states'].columns:
            
            #format plot & pdf labels
            name1= ''.join(column) #HC: convert tuple to str
            name2 = ''.join([char for char in name1 if char != '/']) #HC: remove potential '/' from metabolite name
            name3 = ''.join([char for char in strain if char != '/']) #HC: remove potential '/' from strain name
            na=name2+'_'+name3
            # Plot
            plt.figure()
            ax = plt.gca() #get current axes
            strain_df['states'].reset_index().plot(x='Time',y=column,ax=ax,label=name2)
            trajectory_df.plot(y=column,ax=ax,label='Computed from derivative')                        
            #HC: save as plot pdf            
            plt.savefig('checkDerivatives/' + na + '_derivativeCheck.pdf',transparent=False)
            #HC: to notebook           
            plt.show()
            plt.close()



def learn_dynamics(df,runString,generations=50,population_size=30,verbose=False):
    '''Find system dynamics Time Series Data.
    
    Take in a Data Frame containing time series data 
    and use that to find the dynamics x_dot = f(x,u).
    '''    
    
    #Fit Model
    model = dynamic_model(df) #HC: create model object with data
    model.search(runString,generations=generations,population_size=population_size,verbose=verbose)

    return model



# HC: A MultiOutput Dynamic Model created from TPOT
class dynamic_model(object):
    '''A MultiOutput Dynamic Model created from TPOT'''
    
    #----------------------------------------------------------------------------- model initiation
    def __init__(self,tsdf):
        self.tsdf = tsdf



    #----------------------------------------------- ------------------------------ best ML model search
    def search(self,runString,generations=50,population_size=30,verbose=False):
        

        
        #col9-------------------------------------- sub-method of search: find best model pipeline
        def fit_single_output(row): #row: output
            
            #Verify nature of y data for fitting
            print('\n\n\nlength of current y: ' +str(len(row)))
            print('1st 3 y-values:\n'+str(row[0])+'\n'+str(row[1])+'\n'+str(row[2]))

            tpot = TPOTRegressor(generations=generations, population_size=population_size, mutation_rate=0.9, crossover_rate=0.1,
                                 early_stop=15,verbosity=verbosity,n_jobs=1,memory='auto',warm_start=True) # WARNING: n_jobs=-2 gives error, -1: max CPU number, 1: GPU
            
            # find best pipeline that TPOT discovered using entire training dataset
            fit_model = tpot.fit(X,row).fitted_pipeline_ #CV score: negated MSE by default
            tpot.export(str(row[0])+'_bestPipeline.py')
            return fit_model
        
        #col9-----------------------------------------------------
        if verbose:
            verbosity = 2
        else:
            verbosity = 0
        
        X = self.tsdf[['states','controls']].values # Verify: value of all features at ALL time points

        #col9: apply find best model pipeline: seems to be iterative run of fit_single_output
        ###self.model_df = self.tsdf['derivatives'].apply(fit_single_output).to_frame()

        pipeline = self.tsdf['derivatives'].apply(fit_single_output)
        
        self.pipeline=pipeline
        self.model_df = pipeline.to_frame()
        self.model_df.to_csv(runString + '_self-model_df.csv')



    #-----------------------------------------------fit model parameters, different from tpot.fit()
    def fit(self,tsdf):
        '''Fit the Dynamical System Model.
        
        Fit the dynamical system model and
        return the map f.
        '''
        
        #update the data frame
        self.tsdf = tsdf
        X = self.tsdf[['states','controls']].values
        
        #Fit the dataframe data to existing models
        #self.model_df.apply(lambda model: print(model),axis=1)
        self.model_df = self.model_df.apply(lambda model: model[0].fit(X,self.tsdf['derivatives'][model.name]),axis=1)



    #----------------------------------------------------predict output based on input into ML model
    def predict(self,X):
        '''Return a Prediction'''
        y = self.model_df.apply(lambda model: model[0].predict(X.reshape(1,-1)),axis=1).values.reshape(-1,)
        return y 



    #------------------------------------------ #HC: empty-----------------------report quality of fit
    def fit_report(self):
        #Calculate The Error Distribution, Broken down by Fit
        pass



# New Stiff Integrator; HC: essentially same as the one in helper.py
def odeintz(fun,y0,times,tolerance=1e-4,verbose=False):
    '''Stiff Integrator for Integrating over Machine Learning Models'''
    maxDelta = 10
    
    f = lambda t,x: fun(x,t)
    r = ode(f).set_integrator('dop853',nsteps=1000,atol=1e-4)
    r.set_initial_value(y0,times[0])

    #progress bar
    #f = FloatProgress(min=0, max=max(times))
    #display(f)

    #Perform Integration
    x = [y0,]
    curTime = times[0]
    for nextTime in times[1:]:
        #print(r.t)
        #while r.successful() and r.t < nextTime:
        while r.t < nextTime:
            if nextTime-curTime < maxDelta:
                dt = nextTime-curTime
            else:
                dt = maxDelta
                
            value = r.integrate(r.t + dt)
            curTime = r.t
            if verbose:
                print(curTime, end='\r')
                sleep(0.001)
            f.value = curTime
        x.append(value)
    return x



def simulate_dynamics(model,strain_df,time_points=None,tolerance=1e-4,verbose=False):
    '''Use Learned Dynamics to Generate a Simulated Trajectory in the State Space'''    
    
    times = strain_df.index.get_level_values(1)
    
    #Get Controls as a Function of Time Using Interpolations
    u_df = strain_df['controls'].apply(lambda y: interp1d(times,y,fill_value='extrapolate'))
    u = lambda t: u_df.apply(lambda x: x(t)).values
    
    #Get Initial Conditions from the Strain Data Frame
    x0 = strain_df['states'].iloc[0].values
    
    #Solve Differential Equation For Same Time Points
    #f = lambda x,t: (model.predict(np.concatenate([x, u(t)])),print(t))[0]
    f = lambda x,t: model.predict(np.concatenate([x, u(t)]))
    
    #Return DataFrame with Predicted Trajectories (Use Integrator with Sufficiently Low tolerances...)
    sol = odeintz(f,x0,times,tolerance=tolerance)
    #sol = odeint(f,x0,times,atol=5*10**-4,rtol=10**-6)
    trajectory_df = pd.DataFrame(sol,columns=strain_df['states'].columns)
    trajectory_df['Time'] = times
    #strain_df.info() #HC: get column data format & size
    #strain_df.describe() #HC: column data statistics,e.g. mean, 25%, etc
    
    return trajectory_df