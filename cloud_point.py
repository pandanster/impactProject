import numpy as np
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics

def getTimeDiff(date1,date2):
	date1=date1/1000
	date2=date2/1000
	date1=datetime.datetime.fromtimestamp(date1)
	date2=datetime.datetime.fromtimestamp(date2)
	delta=date2-date1
	return divmod(delta.total_seconds(),60)[1]
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

def cloudPointCluster(inFile,inProximity,pointCount,distance,pltTitle,pltType='3d',saveDir=None,axis=None):
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
	colors=['red','green','blue','orange','black','yellow','pink','brown','cyan','purple']
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) > pointCount:
				fig=plt.figure()
				fig.suptitle(pltTitle,fontsize=20)
				if pltType=='3d':
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
						if label !=-1:
							ax.scatter(x_[:,0],x_[:,1],x_[:,2],c=colors[u],marker='o')
					plt.show()
				else:
					ax=fig.add_subplot(111)

					if axis=='xy' or axis == None:
						ax.set_xlabel('x')
						ax.set_ylabel('y')
						x=np.array([[x,y] for x,y in zip(data_x,data_y)])
					elif axis == 'yz':
						ax.set_xlabel('y')
						ax.set_ylabel('z')
						x=np.array([[x,y] for x,y in zip(data_y,data_z)])
					elif axis == 'xz':
						ax.set_xlabel('x')
						ax.set_ylabel('z')
						x=np.array([[x,y] for x,y in zip(data_x,data_z)])
					x=x.reshape(-1,2)
					cluster=db.fit(x)
					pointLabels=cluster.labels_
					labels=list(set(cluster.labels_))
					for u,label in enumerate(labels):	
						indices=[i for i,j in enumerate(pointLabels) if j == label]
						x_=x[indices]	
						if label !=-1:
							ax.scatter(x_[:,0],x_[:,1],c=colors[u],marker='o')

					plt.savefig(saveDir+'/'+'Image-'+str(count)+'.jpeg')
					count+=1
					plt.close()
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			if line[0] <= 2:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
				data_inten.append(line[3])

def intensGroups(data):
	data=[(u,v) for u,v in enumerate(data)]
	srtdData=sorted(data,key=lambda x:x[1])
	maxVal=max(data,key=lambda x:x[1])
	intenDict={}
	for j in range(0,int(maxVal[1]),5):
		intenDict[j]=[]
	intenDict[j]=[]
	for val in srtdData:
		try:
			intenDict[int(val[1]/5)*5].append(val[0])		 
		except:
			print(int(val[1]/5),val)
			exit(0)
	return intenDict	
	
'''
Takes four parameters
@param inFile - The input File that contains the captured data
@param inProximity - Decides the time for which cloud Points are treated as cooccuring, setting as zero treats only cloud points with same timestamp as cooccuring
@param pointCount - Decides the point count - number of points in the current proximity- threshold to display
@param pltTitle - Sets the title for the plot
@return None
Displays the 3D projections of the points
'''

def cloudPointVisualizeInten(inFile,inProximity,pointCount,pltTitle,intensityThreshold):
	inFile=open(inFile,'r')
	lines=inFile.readlines()
	lines=[line.strip().split(' ') for line in lines]
	data_x=[]
	data_y=[]
	data_z=[]
	data_inten=[]
	currTime=None
	count=0
	colors=['red','green','blue','orange','black','yellow','pink','brown','cyan','purple']
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) > pointCount:

				fig=plt.figure()
				fig.suptitle(pltTitle,fontsize=20)
				ax=fig.add_subplot(111,projection='3d')
				ax.set_xlabel('x')
				ax.set_ylabel('y')
				ax.set_zlabel('z')
				intenDict=intensGroups(data_inten)
				col_count=0
				for key in intenDict.keys():
					if len(intenDict[key]) >0:
						print(key,colors[col_count])
						temp_x=np.array(data_x)[intenDict[key]].tolist()							
						temp_y=np.array(data_y)[intenDict[key]].tolist()							
						temp_z=np.array(data_z)[intenDict[key]].tolist()							
						ax.scatter(temp_x,temp_y,temp_z,c=colors[col_count],marker='o')
					col_count+=1
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

def cloudPointVisualize(inFile,inProximity,pointCount,pltTitle,saveDir):
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
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) > pointCount:
				plt.clf()
				fig=plt.figure()
				fig.suptitle(pltTitle,fontsize=20)
				ax=fig.add_subplot(111)
				ax.set_xlabel('x')
				ax.set_ylabel('y')
#				ax.set_zlabel('z')
				ax.scatter(data_x,data_y,c='r',marker='o')
				plt.savefig(saveDir+'/'+'Image-'+str(count)+'.jpeg')
				plt.close()
				count+=1
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			if line[0] <= 2:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
				data_inten.append(line[3])

saveDir='/home/gmuadmin/Desktop/Research Experimets/ti-dataacq/riley_rain_fast_cluster_xz'
#cloudPointVisualize('48cm_exp/Frederick/frederick_push_fast_01_exp_04_06_2019.txt',0.2,0,'Gesture',saveDir)
cloudPointCluster('48cm_exp/Riley/riley_rain_fast_05_exp_04_06_2019.txt',0.2,0,0.2,'Gesture','2d',saveDir,'xz')
