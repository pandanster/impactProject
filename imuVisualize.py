import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from scipy.signal import spectrogram, iirfilter,freqz,decimate,filtfilt,correlate

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
			else:
				ax.plot(inFile[:,i],label='x')
				ax.plot(inFile[:,i+1],label='y')
				ax.plot(inFile[:,i+2],label='z')
			plt.legend()
			plt.show()
#sensorDataVisualize('Sensor_record_20190515_161127_AndroSensor.csv',plotType='3D',smooth=True,window=10)
b1,a1=build_filter(6,1000,None,'lowpass',8000,'ellip',.01,None)
