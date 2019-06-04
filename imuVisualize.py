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

def FilterData():
	
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
x=[1,2,3,4,5,6,6,7,7,2,2,5,2,4]
#print(movingAvg(x,2))			
sensorDataVisualize('Sensor_record_20190515_161127_AndroSensor.csv',plotType='3D',smooth=True,window=10)
