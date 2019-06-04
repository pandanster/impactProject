import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics

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

def cloudPointCluster(inFile,inProximity,pointCount,distance,pltTitle):
	inFile=open(inFile,'r')
	lines=inFile.readlines()
	lines=[line.strip().split(' ') for line in lines]
	data_x=[]
	data_y=[]
	data_z=[]
	data_inten=[]
	currTime=None
	count=0
	db=DBSCAN(eps=distance)
	colors=['red','green','blue','orange','black','yellow']
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif line[-1] > currTime+inProximity:
			if len(data_x) > pointCount:
				fig=plt.figure()
				fig.suptitle(pltTitle,fontsize=20)
				ax=fig.add_subplot(111,projection='3d')
				ax.set_xlabel('x')
				ax.set_ylabel('y')
				ax.set_zlabel('z')
				x=np.array([data_x,data_y,data_z])
				x=x.reshape(-1,3)
				cluster=db.fit(x)
				pointLabels=cluster.labels_
				labels=list(set(cluster.labels_))
				for u,label in enumerate(labels):	
					indices=[i for i,j in enumerate(pointLabels) if j == label]
					x_=x[indices]	
					ax.scatter(x_[:,0],x_[:,1],x_[:,2],c=colors[u],marker='o')
				plt.show()
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			data_x.append(line[0])
			data_y.append(line[1])
			data_z.append(line[2])
			data_inten.append(line[3])
'''
Takes four parameters
@param inFile - The input File that contains the captured data
@param inProximity - Decides the time for which cloud Points are treated as cooccuring, setting as zero treats only cloud points with same timestamp as cooccuring
@param pointCount - Decides the point count - number of points in the current proximity- threshold to display
@param pltTitle - Sets the title for the plot
@return None
Displays the 3D projections of the points
'''

def cloudPointVisualize(inFile,inProximity,pointCount,pltTitle):
	inFile=open(inFile,'r')
	lines=inFile.readlines()
	lines=[line.strip().split(' ') for line in lines]
	data_x=[]
	data_y=[]
	data_z=[]
	data_inten=[]
	currTime=None
	count=0
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif line[-1] > currTime+inProximity:
			if len(data_x) > pointCount:
				fig=plt.figure()
				fig.suptitle(pltTitle,fontsize=20)
				ax=fig.add_subplot(111,projection='3d')
				ax.set_xlabel('x')
				ax.set_ylabel('y')
				ax.set_zlabel('z')
				ax.scatter(data_x,data_y,data_z,c='r',marker='o')
				plt.show()
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			data_x.append(line[0])
			data_y.append(line[1])
			data_z.append(line[2])

#cloudPointVisualize('temp',0,60,'Gesture')
cloudPointCluster('temp',0,60,0.5,'Gesture')
