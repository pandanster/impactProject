import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from scipy.signal import spectrogram, iirfilter,freqz,decimate,filtfilt,correlate
from sklearn import tree 
import glob
import io 
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

'''
Computes the mean of the data sample
@input array of any dimension
returns the flat mean of the given array
'''
def getMean(data):
	result = np.mean(data)
	return result

'''
Computes the mean of the X, Y, and z axis of the IMU
@datax array corresponding to the x axis of the data
@datay array corresponding to the y axis of the data
@dataz array corresponding to the z axis of the array
returns the mean over the summation of the x, y, and z axis
'''
def getTotalMean(datax, datay, dataz):
	data = datax + datay + dataz
	result = np.mean(data)
	return result
'''
Computes the area under a curve, define as the sum of values under the curve
@data array of any dimension
returns the sum of the array
'''
def getArea(data):
	result = np.sum(data)
	return result 
'''
Computes the distribution that is the range of data along the three axis
@datax data corresponding to the x axis 
@datay data corresponding to the y axis
@dataz data corresponding to the z axis
returns the distribution along the three axis 
'''
def getPostureDist(datax, datay, dataz):
	diffxy = datax - datay
	diffxz = datax - dataz
	diffyz = datay - dataz
	return [diffxy, diffxz, diffyz]
'''
Computes the absolute mean
@data input array of any dimension
returns the absolute mean on the flattened data 
'''
def getAbsMean(data):
	result = np.mean(np.abs(data))
	return result
'''
Computes the absolute area under the curve
@data array of any dimension
returns the absolute area under the curve
'''
def getAbsArea(data):
	result = np.sum(np.abs(data))
	return result	

'''
Computes Moving average Takes two paramters
@x Sequence to compute moving average on
@N The window size
return the smoothed sequence
'''
def movingAvg(x,N):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / float(N)

'''
Applies a filter for the given sequence takes three parameters
@b filer coeffiecient
@a filter coefficient
@data The data to be filtered
returns the filtered Data
'''

def FilterData(b,a,data):
	return filtfilt(b,a,data)	
'''
Function to build a digital filter
Takes the following parameters
@Order of the filter to be designed
@pass_band the pass band frequency
@stop_band the stop band frequency
@fs - sampling rate of the signal Number of samples per second
@ripple - How much is the tolearble ripple in the pass band  
@attenuation - How much is the tolerable ripple in the stop band if None will generate plots for different attenuation values
'''
def build_filter(Order,pass_band,stop_band,band,fs,filter,ripple,attenuation,plot=False):
	nyq=fs/2
	if band== 'bandpass':
		pass_low=pass_band/nyq
		pass_high=stop_band/nyq
		stop_low=pass_low*0.8
		stop_high=pass_high/0.8
		wn=[pass_low,pass_high]
	elif band== 'lowpass': 
		wn=pass_band/nyq
	elif band=='highpass':
		wn=pass_band/nyq
	else:
		return None
	if attenuation!=None:
		b,a = iirfilter(Order,Wn=wn,btype=band,rp=ripple,rs=attenuation,ftype=filter)
		w,h =freqz(b,a)
		if plot==True:
			plt.plot((nyq/np.pi)*w,abs(h))
			plt.title(filter+' filter frequency response')
			plt.xlabel('Frequency')
			plt.ylabel('Amplitude')
			plt.grid(True)
			plt.legend(loc='best')
			plt.show()
	elif Order !=None:
		for i in [10,30,40,60,80]:
			b,a = iirfilter(Order,Wn=wn,btype=band,rp=.05,rs=i,ftype=filter)
			w,h =freqz(b,a)
			plt.plot((nyq/np.pi)*w,abs(h),label='stop band attenuation= %d' % i)
		plt.title(filter+' filter frequency response')
		plt.xlabel('Frequency')
		plt.ylabel('Amplitude')
		plt.grid(True)
		plt.legend(loc='best')
		plt.show()
	elif attenuation==None:
		for i in [2,4,6]:
			b,a = iirfilter(i,Wn=wn,btype=band,rp=.01,rs=40,ftype=filter)
			w,h =freqz(b,a)
			plt.plot((nyq/np.pi)*w,abs(h),label='Order= %d' % i)
		plt.title(filter+' filter frequency response')
		plt.xlabel('Frequency')
		plt.ylabel('Amplitude')
		plt.grid(True)
		plt.legend(loc='best')
		plt.show()
	return (b,a)
		
'''
Builds a decision tree classifer using the training data and labels for the given depth
@trainData features for the training data
@trainLabels integer labels corresponding the training data
@depth Depth of the tree
returns the decision tree
'''
def buildTree(trainData,trainLabels,depth):
	dtc= DecisionTreeClassifier(max_depth=depth)
	dtc.fit(trainData,trainLabels)
	return dtc

'''
Can be used to print the decision tree
@dtree The decision Tree built using buildTree
@features Feature names like area, mean, etc.
Saves an image file in the current directory corresponding to the built tree
'''
def printTree(dtree,features):
	dotfile=io.StringIO()
	export_graphviz(decision_tree=dtree,feature_names=features,out_file=dotfile,class_names=['alarm','call'])
	graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
	graph.write_png("dtree.png")

'''
Create Features for the inputFile
@inFile the input file name
returns two features Area and Mean
'''

def getFeatures(inFile):
	inFile=open(inFile,'r')
	inFile=inFile.readlines()
	inFile=inFile[1:]
	inFile=[f.strip().split(',') for f in inFile]
	inFile=[[float(x) for x in f[:-1]] for f in inFile]
	inFile=np.array(inFile)
	return getArea(inFile),getMean(inFile)
'''
Takes the two training and test Directories, builds a decision Tree classifier using training Files and predicts on the testing File
@trainDir Directory containging the training files
@testDir Directory containing the test files
You can control the depth of the tree using the depth parameter to the function call buildTree
prints the predictions for the test files
'''
def trainAndPredict(trainDir,testDir):
	trainLabels=[]
	testLabels=[]
	trainData=[]
	testData=[]
	labels=['alarm','call'] #labels for your classes
	trainFiles=glob.glob(trainDir+'/*')				
	testFiles=glob.glob(testDir+'/*')
	for trainFile in trainFiles:
		feat1,feat2=getFeatures(trainFile) # Obtain the features for the current input File
		trainData.append([feat1,feat2])
		if 'alarm' in trainFile:
			trainLabels.append(labels.index('alarm')) # The labels should be integers for the classifier, so we convert the strin label to its corresponding list index 
		elif 'call' in trainFile:
			trainLabels.append(labels.index('call'))
	for testFile in testFiles:
		feat1,feat2=getFeatures(testFile)
		testData.append([feat1,feat2])
		if 'alarm' in testFile:
			testLabels.append(labels.index('alarm'))
		elif 'call' in testFile:
			testLabels.append(labels.index('call'))
	dtc=buildTree(trainData,trainLabels,depth=1) #Build the decision tree of depth 1
	#printTree(dtc,['Area','Mean'])
	print("These are the predicted Labels {}".format([labels[int(x)] for x in list(dtc.predict(testData))])) #Convert the predicted integer classes to their corresponding string labels
	print("These are the true Labels {}".format([labels[int(x)] for x in list(testLabels)]))
	exit(0)
'''
Takes five parameters
@param inFile - The input File that contains the captured data
@param inProximity - Decides the time for which cloud Points are treated as cooccuring, setting as zero treats only cloud points with same timestamp as cooccuring
@param pointCount - Decides the point count - number of points in the current proximity- threshold to display
@param distance - Decides the distance between points to treat them as a cluster
@param pltTitle - Sets the title for the plot
@return None
Displays the 3D projections of the points
'''

def sensorDataVisualize(inFile,plotType=None,smooth=False,window=None):
	inFile=open(inFile,'r')
	inFile=inFile.readlines()
	inFile=inFile[1:]
	inFile=[f.strip().split(',') for f in inFile]
	inFile=[[float(x) for x in f[:-1]] for f in inFile]
	inFile=np.array(inFile)
	titles=['Accelerometer','Linear Accelerometer','Gyroscope']
	titleCount=0
	for i in range(0,9,3):
		fig=plt.figure()
		fig.suptitle(titles[titleCount],fontsize=20)
		titleCount+=1
		if plotType=='3D':
			ax=fig.add_subplot(111,projection='3d')
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('z')
			if smooth:
				ax.plot(movingAvg(inFile[:,i].tolist(),window),movingAvg(inFile[:,i+1].tolist(),window),movingAvg(inFile[:,i+2].tolist(),window))
			else:
				ax.plot(inFile[:,i],inFile[:,i+1],inFile[:,i+2])
			plt.show()
		else:
			ax=fig.add_subplot(111)
			if smooth:
				ax.plot(movingAvg(inFile[:,i].tolist(),window),label='x')
				ax.plot(movingAvg(inFile[:,i+1].tolist(),window),label='y')
				ax.plot(movingAvg(inFile[:,i+2].tolist(),window),label='z')
				ax.grid()
			else:
				ax.plot(inFile[:,i],label='x')
				ax.plot(inFile[:,i+1],label='y')
				ax.plot(inFile[:,i+2],label='z')
				ax.grid()
			plt.legend()
			plt.show()

'''
Example for filtering Data
Assumed sampling rate is 100 Hz
Filter order is 6, low pass with frequency of 2 Hz ripple .01 and 
stop band attenuation of 30
The parameter 2 is the frequency for which you are filtering
The parameter 5 is your sampling rate
'''
def FilterExample(inFile):
	b1,a1=build_filter(8,5,None,'lowpass',100,'ellip',.01,30)
	inFile=open(inFile,'r')
	inFile=inFile.readlines()
	inFile=inFile[1:]
	inFile=[f.strip().split(',') for f in inFile]
	inFile=[[float(x) for x in f[:-1]] for f in inFile]
	inFile=np.array(inFile)
	plt.plot(inFile[:,0],label='x')
	plt.plot(inFile[:,1],label='y')
	plt.plot(inFile[:,2],label='z')
	plt.title('Before Filter')
	plt.legend()
	plt.show()
	plt.plot(FilterData(b1,a1,inFile[:,0]),label='x')
	plt.plot(FilterData(b1,a1,inFile[:,1]),label='y')
	plt.plot(FilterData(b1,a1,inFile[:,2]),label='z')
	plt.title('Before Filter')
	plt.legend()
	plt.show()

trainAndPredict('./data_dir/train/','./data_dir/test/')			
#sensorDataVisualize('Sensor_record_20190515_161127_AndroSensor.csv',plotType=None,smooth=True,window=200)
# FilterExample('Sensor_record_20190515_161127_AndroSensor.csv')
