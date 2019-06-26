import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import ConvexHull
from scipy.special import gammainc
from  sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
import random
import copy
import math
import multiprocessing as mp
from queue import Queue
from scipy.spatial import Delaunay
import logging,sys
logging.basicConfig(stream=sys.stderr,level=logging.INFO)
logger=logging.getLogger('cloudPoint')
'''
Computes Moving average Takes two paramters
@x Sequence to compute moving average on
@N The window size
return the smoothed sequence
'''
def movingAvg(x,N,axis=None):
	if axis is None:
		cumsum = np.cumsum(np.insert(x, 0, 0))
	else:
		cumsum = np.cumsum(np.insert(x, 0, 0),axis=0)
			
	return (cumsum[N:] - cumsum[:-N]) / float(N)

def getUniformPoints(center,radius,NPoints):
	x=np.random.normal(size=(NPoints,center.shape[0]))
	sq=np.sum(x**2,axis=1)
	nm=radius*gammainc(center.shape[0]/2,sq/2)**(1/center.shape[0])/np.sqrt(sq)
	nm_reshaped=np.tile(nm.reshape(NPoints,1),(1,center.shape[0]))
	points=center+np.multiply(x,nm_reshaped)
	return points
		
def getCentroid(Cluster):
	return np.mean(Cluster)
		
class particle:
	def __init__(self,x,y,z,noise):
		self.x=x
		self.y=y
		self.z=z
		self.w=0
		self.noise=noise

	def setNoise(self,noise):
		self.noise=noise
class measure:
	def __init__(self):
		self.x=0
		self.y=0
		self.z=0

	def setValue(self,x,y,z):
		self.x=x
		self.y=y
		self.z=z

class Particles:
	def __init__(self,particles):
		self.particles=particles
		for particle in self.particles:
			particle.w=(1/len(self.particles))
		
	def predict(self,dt):		
		for particle in self.particles:
			particle.x=particle.x+dt[0]+random.gauss(0,particle.noise)
			particle.y=particle.y+dt[1]+random.gauss(0,particle.noise)
			particle.z=particle.z+dt[2]+random.gauss(0,particle.noise)

	def importance(self,mu,sigma,x):
		return multivariate_normal.pdf(x,mu,sigma)

	def update(self,measures,measureNoise,):	
		for particle in self.particles:
			w=0
			for measure in measures:
				try:
					w+=math.log(self.importance(np.array([measure[0],measure[1],measure[2]]),measureNoise,np.array([particle.x,particle.x,particle.y])))		
				except:
					w+=math.log(1e-20)		
			particle.w=w
	def normWeights(self):
		totalW=0
		for particle in self.particles:
			totalW+=particle.w
		for particle in self.particles:
			particle.w/=totalW

	def getMaxWeight(self):
		weights=[particle.w for particle in self.particles]
		return max(weights)

	def getArray(self):
		particles=[[particle.x,particle.y,particle.z] for particle in self.particles]
		return np.array(particles)

	def resampleParticles(self):
		indices=[int(random.uniform(0,len(self.particles))) for i in range(len(self.particles))]
		beta=0
		maxW=self.getMaxWeight()
		particles=[]
		for i in range(len(self.particles)):
			beta+=random.uniform(0,2*maxW)
			index=indices[i]
			while self.particles[index].w < beta:
				beta=beta-self.particles[index].w
				index+=1
				if index==len(self.particles):
					index=0
			particles.append(copy.deepcopy(self.particles[index]))
		self.particles=particles

	def plotWeights(self):
		weight=[particle.w for particle in self.particles]
		plt.plot(weight)
		plt.show()

	def plot(self,pltTitle):
		particles=self.getArray()
		fig=plt.figure()
		fig.suptitle(pltTitle,fontsize=20)
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_xlim(0,1.5)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		ax.scatter(particles[:,0],particles[:,1],particles[:,2],c='r',marker='o')
		plt.show()

	def save(self,pltTitle,saveDir,axis):
		particles=self.getArray()
		plt.clf()
		fig=plt.figure()
		ax=fig.add_subplot(111)

		
		if axis=='xy':
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_xlim(-0,1.6)
			ax.set_ylim(-1,1)
			ax.scatter(particles[:,0],particles[:,1],c='r',marker='o')
		elif axis=='yz':
			ax.set_xlabel('y')
			ax.set_ylabel('z')
			ax.set_xlim(-2,2)
			ax.set_ylim(-2,2)
			ax.scatter(particles[:,1],particles[:,2],c='r',marker='o')
		else:
			ax.set_xlabel('x')
			ax.set_ylabel('z')
			ax.set_xlim(-2,2)
			ax.set_ylim(-2,2)
			ax.scatter(particles[:,0],particles[:,2],c='r',marker='o')
		plt.savefig(saveDir+'/'+pltTitle+'.jpeg')
		plt.close()
		
def getTimeDiff(date1,date2):
	date1=date1/1000
	date2=date2/1000
	date1=datetime.datetime.fromtimestamp(date1)
	date2=datetime.datetime.fromtimestamp(date2)
	delta=date2-date1
	return divmod(delta.total_seconds(),60)[1]

def getConvexHull(points):
	pts=np.array([[0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 1, 0],[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], ])
	hull=ConvexHull(pts)
	print(hull.area)
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")
	for s in hull.simplices:
		s=np.append(s,s[0])
		ax.plot(pts[s,0],pts[s,1],pts[s,2],"r-")
	plt.show()
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
	#db=KMeans(n_clusters=3,random_state=0)
	db=DBSCAN(eps=distance)
	colors=['red','green','blue','orange','black','yellow','pink','brown','cyan','purple','grey','violet']
	clusters=[]
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) > pointCount:
				data=np.array([[x,y,z] for x,y,z in zip(data_x,data_y,data_z)])
				if len(clusters) > 2:
					x=None
					for clust in clusters:
						if x is None:
							x=clust
						else:
							x=np.concatenate((x,clust),axis=0)		
					x=x.reshape(-1,3)
					cluster=db.fit(x)
					pointLabels=cluster.labels_
					labels=list(set(cluster.labels_))
					fig=plt.figure()
					fig.suptitle(pltTitle,fontsize=20)
					if pltType=='3d':
						ax=fig.add_subplot(111,projection='3d')
						ax.set_xlabel('x')
						ax.set_ylabel('y')
						ax.set_zlabel('z')
						ax.set_xlim(0,1.5)
						ax.set_ylim(-1,1)
						ax.set_zlim(-1,1)
	
						for u,label in enumerate(labels):	
							indices=[i for i,j in enumerate(pointLabels) if j == label]
							x_=x[indices]	
						#if label !=-1:
							ax.scatter(x_[:,0],x_[:,1],x_[:,2],c=colors[u],marker='o')
						plt.show()
					else:
						ax=fig.add_subplot(111)

						if axis=='xy' or axis == None:
							ax.set_xlabel('x')
							ax.set_ylabel('y')
							ax.set_xlim(0,1.5)
							ax.set_ylim(-1,1)
							i=0
							j=1
					#	x=np.array([[x,y] for x,y in zip(data_x,data_y)])
						elif axis == 'yz':
							ax.set_xlabel('y')
							ax.set_ylabel('z')
							ax.set_xlim(-1,1)
							ax.set_ylim(-1,1)
					#	x=np.array([[x,y] for x,y in zip(data_y,data_z)])
							i=1
							j=2
						elif axis == 'xz':
							ax.set_xlabel('x')
							ax.set_ylabel('z')
							ax.set_xlim(0,1.5)
							ax.set_ylim(-1,1)
							i=0
							j=2
					#	x=np.array([[x,y] for x,y in zip(data_x,data_z)])
					
					#x=x.reshape(-1,2)
				#	cluster=db.fit(x)
				#	pointLabels=cluster.labels_
				#	labels=list(set(cluster.labels_))
						for u,label in enumerate(labels):	
							indices=[i for i,j in enumerate(pointLabels) if j == label]
							x_=x[indices]	
							if label !=-1:
								try:
									ax.scatter(x_[:,i],x_[:,j],c=colors[u],marker='o')
								except Exception as e:
									print(e)
									print(u,currTime)
						if count==0:
							try:
								os.mkdir(saveDir+'/'+str(distance)+axis)
							except:
								pass
						plt.savefig(saveDir+'/'+str(distance)+axis+'/Image-'+str(count)+'.jpeg')
						count+=1
						plt.close()
					clusters.pop(0)
					clusters.append(data)
				else:
					clusters.append(data)
				data_x=[]
				data_y=[]
				data_z=[]
				data_inten=[]
				currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[0] >= 0.3 and line[1] <= 0.6 and line[1] >=-0.6 and line[2]<=0.6 and line[2]>=-0.6:
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

def cloudPointVisualize(inFile,inProximity,pointCount,pltTitle,saveDir=None,pltType='3d',axis=None):
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
				#plt.clf()
				fig=plt.figure()
				fig.suptitle(pltTitle,fontsize=20)
				if pltType=='3d':
					ax=fig.add_subplot(111,projection='3d')
					ax.set_xlim(0,1.5)
					ax.set_ylim(-1,1)
					ax.set_zlim(-1,1)
					ax.set_xlabel('x')
					ax.set_ylabel('y')
					ax.set_zlabel('z')
					ax.scatter(data_x,data_y,data_z,c='r',marker='o')
				else:
					ax=fig.add_subplot(111)
					if axis=='xy':
						ax.set_xlabel('x')
						ax.set_ylabel('y')
						ax.set_xlim(0,1.5)
						ax.set_ylim(-1,1)
						ax.scatter(data_x,data_y,c='r',marker='o')
					elif axis=='yz':
						ax.set_xlabel('y')
						ax.set_ylabel('z')
						ax.set_xlim(-1,1)
						ax.set_ylim(-1,1)
						ax.scatter(data_y,data_z,c='r',marker='o')
					else:
						ax.set_xlabel('x')
						ax.set_ylabel('z')
						ax.set_xlim(0,1.5)
						ax.set_ylim(-1,1)
						ax.scatter(data_x,data_z,c='r',marker='o')
						
				if saveDir is None:
					plt.show()
				else:
					plt.savefig(saveDir+'/'+'Image-'+str(count)+'.jpeg')
					plt.close()
				count+=1
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[1] <=1 and line[1] >=-1 and line[2]<=1 and line[2]>=-1:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
				data_inten.append(line[3])

def processFiles(inDir,saveDir):
	inFiles=glob.glob(inDir+'/*')
	planes=['xy','yz','xz']
	for inFile in inFiles:
		name=inFile.strip().split('/')[-1]
		dirName=name.split('_')[1]+name.split('_')[2]
		os.mkdir(saveDir+'/'+dirName)
		for plane in planes:
			try:
				os.mkdir(saveDir+'/'+dirName+'/'+plane)
			except:
				print("Exists:"+dirName)
			try:
				os.mkdir(saveDir+'/'+dirName+'/'+plane+'-clust')
			except:
				print("Exists:"+dirName)
			cloudPointVisualize(inFile,0.2,0,dirName,saveDir+'/'+dirName+'/'+plane,'2d',plane)
			cloudPointCluster(inFile,0.2,0,0.1,dirName,'2d',saveDir+'/'+dirName+'/'+plane+'-clust',axis=plane)

def getDist(x1,x2):
	if type(x1) is list:
		x1=np.array(x1)
	if type(x2) is list:
		x2=np.array(x2)
	return np.sum((x1-x2)**2)

def getClusters(x,dist):
	db=DBSCAN(eps=dist)
	cluster=db.fit(x)
	pointLabels=cluster.labels_
	labels=list(set(cluster.labels_))
	culsterDist=0
	clusters=[]
	centroids=[]
	for u,label in enumerate(labels):	
		indices=[i for i,j in enumerate(pointLabels) if j == label]
		x_=x[indices]	
		if label !=-1:
			clusters.append(x_)
			centroids.append(np.mean(x_,axis=0))
	return clusters,centroids

def saveFigure(x,pltTitle='temp',clustered=False,saveDir=None,pltType='3d',axis=None):
	plt.clf()
	fig=plt.figure()
	fig.suptitle(pltTitle,fontsize=20)
	colors=['red','green','blue','orange','black','yellow','pink','brown','cyan','purple','grey','violet']
	if pltType=='3d':
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_xlim(0,1.5)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		if clustered:
			for u,x_ in enumerate(x):	
				print(u,x_)
				ax.scatter(x_[:,0],x_[:,1],x_[:,2],c=colors[u],marker='o')
		else:
			ax.scatter(x[:,0],x[:,1],x[:,2],c='r',marker='o')
		plt.show()
	else:
		ax=fig.add_subplot(111)
		if axis=='xy' or axis == None:
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_xlim(0,1.5)
			ax.set_ylim(-1,1)
			i=0
			j=1
		elif axis == 'yz':
			ax.set_xlabel('y')
			ax.set_ylabel('z')
			ax.set_xlim(-1,1)
			ax.set_ylim(-1,1)
			i=1
			j=2
		elif axis == 'xz':
			ax.set_xlabel('x')
			ax.set_ylabel('z')
			ax.set_xlim(0,1.5)
			ax.set_ylim(-1,1)
			i=0
			j=2
		if clustered:
			for u,x_ in enumerate(x):	
				ax.scatter(x_[:,i],x_[:,j],c=colors[u],marker='o')
		else:
			ax.scatter(x[:,0],x[:,1],x[:,2],c='r',marker='o')
		if saveDir is None:
			plt.show()
		else:
			plt.savefig(saveDir+'/'+pltTitle+'.jpeg')
			plt.close()
					
def getHandClusters(x,dist,centroid,hclusters):
	db=DBSCAN(eps=dist,min_samples=5)
	cluster=db.fit(x)
	pointLabels=cluster.labels_
	labels=list(set(cluster.labels_))
	maxCluster=[]
	culsterDist=0
	mean=[]
	for u,label in enumerate(labels):	
		indices=[i for i,j in enumerate(pointLabels) if j == label]
		x_=x[indices]	
		if label !=-1:
			clusterDist=getDist(np.mean(x_,axis=0),centroid)
			if clusterDist!=0 and math.sqrt(clusterDist) >=0.2:
				maxCluster.append(x_)
				mean.append(np.mean(x_,axis=0))
	return maxCluster,mean

def getClusterPointOverlap(cluster1,cluster2):
	temp2=None
	temp1=None
	if len(cluster1) == 0 or len(cluster2) ==0:
		return 0
	for cluster in cluster2:
		if temp2 is None:
			temp2=cluster
		else:
			temp2=np.concatenate((temp2,cluster),axis=0)
	for cluster in cluster1:
		if temp1 is None:
			temp1=cluster
		else:
			temp1=np.concatenate((temp1,cluster),axis=0)
	cluster1=np.array(temp1,dtype=float)
	cluster2=np.array(temp2,dtype=float)
	t=np.isclose(cluster1[:,None],cluster2,.00001,1e-8).all(-1)
	overlap=np.where(t)
	logger.debug("Cluster1: %s,Cluste2: %s, Overlap:%s",cluster1.shape[0],cluster2.shape[0],overlap[0].shape[0])
	return overlap[0].shape[0]/cluster1.shape[0]

def getClusterAreaOverlap(cluster1,cluster2,count):
	temp2=None
	temp1=None
	if len(cluster1) == 0 or len(cluster2) ==0:
		return 0
	for cluster in cluster2:
		if temp2 is None:
			temp2=cluster
		else:
			temp2=np.concatenate((temp2,cluster),axis=0)
	for cluster in cluster1:
		if temp1 is None:
			temp1=cluster
		else:
			temp1=np.concatenate((temp1,cluster),axis=0)
	cluster1=np.array(temp1,dtype=float)
	cluster2=np.array(temp2,dtype=float)
	mu=np.mean(cluster1,axis=0)
	sigma=np.std(cluster1)
	sigma=.005
	overlaps=[x for x in cluster2.tolist() if multivariate_normal.pdf(x,mu,sigma) > 0.8]
	logger.debug("Overlap is:%s",len(overlaps)/cluster2.shape[0])
	return len(overlaps)/cluster2.shape[0]
		
def handClassifier(clusters,center,radius,delta):
	handIndices=[]
	for i,cluster in enumerate(clusters):
		logger.debug("Distance from center:%s",math.sqrt(getDist(center,np.mean(cluster,axis=0))))
		if math.sqrt(getDist(center,np.mean(cluster,axis=0)) > 0.2) and math.sqrt(getDist(center,np.mean(cluster,axis=0))) < radius:
			handIndices.append(i)
	return handIndices

def getExistingClusters(distClusters,distCentroids,currentCentroids,currClusters,chosenCentroids,chosenClusters,delta,count):
	existingClusters=[]
	for key,distCluster in distClusters.items():
		for i,cluster in enumerate(distCluster):
			logger.debug("Dist::%s,len of clusters:%s,# Points:%s,Distances:%s,Chosen:%s,ChosenCentroid:%s,distCentroid:%s",key,len(distClusters[key]),cluster.shape[0],[math.sqrt(getDist(distCentroids[key][i],cent)) for cent in chosenCentroids],len(chosenCentroids),chosenCentroids,distCentroids[key][i])
			if not any([math.sqrt(getDist(distCentroids[key][i],cent)) < delta for cent in chosenCentroids]):
				for centroid in currentCentroids:
					logger.debug("Distance:%s, existing Overlap: %s, chosen Overlap: %s, centroid Dist%s:",key,getClusterPointOverlap(existingClusters,[cluster]),getClusterPointOverlap(chosenClusters,[cluster]),math.sqrt(getDist(centroid,distCentroids[key][i])))
					if (math.sqrt(getDist(centroid,distCentroids[key][i])) < delta or any([getClusterAreaOverlap([cust],[cluster],count) > 0.7 for cust in currClusters])) and getClusterPointOverlap(existingClusters,[cluster]) < 0.7 and getClusterPointOverlap(chosenClusters,[cluster]) < 0.7  and  all([getClusterAreaOverlap([cust],[cluster],count) < 0.7 for cust in chosenClusters]) and all([getClusterAreaOverlap([cust],[cluster],count) < 0.7 for cust in existingClusters]):
						existingClusters.append(cluster)
	logger.debug("Existing Clusters:%s for count %s",len(existingClusters),count)
	return existingClusters
	
def getClusterVoting(distClusters,distCentroids,distances,delta,count):
	chosenClusters=[]
	chosenDists=[]
	chosenCentroids=[]
	for i in range(0,len(distances)):
		if len(distClusters[distances[i]]) > 0:
			for v in distCentroids[distances[i]]:
				outIndex=distCentroids[distances[i]].index(v)
				tempClusters=[]
				tempDistances=[]
				for j in range(i+1,len(distances)):
					if len(distClusters[distances[j]]) >0:
						for k in distCentroids[distances[j]]:			
							try:
								inIndex=distCentroids[distances[j]].index(k)
							except:
									print(k)
									print("all")
									print(distCentroids[distances[j]])
									exit(0)
							if math.sqrt(getDist(distCentroids[distances[i]][outIndex],distCentroids[distances[j]][inIndex])) <= delta:
								tempClusters.append(distClusters[distances[j]].pop(inIndex))
								tempDistances.append(distances[j])
								distCentroids[distances[j]].pop(inIndex)
				try:
					tempClusters.append(distClusters[distances[i]].pop(outIndex))
					tempDistances.append(distances[i])
					distCentroids[distances[i]].pop(outIndex)
				except:
					print(i,j,outIndex,len(distCentroids[distances[i]]),len(distClusters[distances[i]]))
				tempClustersSorted=sorted(tempClusters,key=lambda x:x.shape[0])
				tempClusterPoints=[cluster.shape[0] for cluster in tempClustersSorted]
				if len(tempClustersSorted) > 1 and min(tempClusterPoints) >=5 and getClusterPointOverlap(chosenClusters,tempClustersSorted) < 0.7:
					chosenDists.append(1/sum(tempDistances))
					chosenClusters.append(tempClustersSorted[int(len(tempClusters)/2)])
					chosenCentroids.append(np.mean(np.array(tempClustersSorted[int(len(tempClusters)/2)]),axis=0).tolist())
	return chosenClusters,chosenCentroids,chosenDists			
	
def chooseClusters(inFile,distances):
	inProximity=0.1
	inFile=open(inFile,'r')
	lines=inFile.readlines()
	lines=[line.strip().split(' ') for line in lines]
	data_x=[]
	data_y=[]
	data_z=[]
	center=np.array([0,0,0.9])
	radius=0.6
	count=0
	delta=0.1
	currTime=None
	currCentroids=[]
	currClusters=[]
	f=open('choosen-clusters','w')
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			x=np.array([[x,y,z] for x,y,z in zip(data_x,data_y,data_z)])
			x=x.reshape(-1,3)
			distCentroids={}
			distClusters={}
			for dist in distances:
				clusters,centroids=getClusters(x,dist)	
				listCentroids=[cent.tolist() for cent in centroids]
				sortedCentroids=sorted([cent.tolist() for cent in centroids],reverse=True)	
				index=[listCentroids.index(cent) for cent in sortedCentroids]
				tempClusters=np.array(clusters)[index]
				distClusters[dist]=[]
				for i in range(tempClusters.shape[0]):
					distClusters[dist].append(tempClusters[i])
				distCentroids[dist]=sortedCentroids
				'''
				handIndices=handClassifier(distClusters[dist],center,delta,radius)
				hands=[distClusters[dist][i] for i in range(len(distClusters[dist])) if i in handIndices]
				other=[distClusters[dist][i] for i in range(len(distClusters[dist])) if i not in handIndices]
				handCluster=None
				for hand in hands:
					if handCluster is None:
						handCluster=hand
					else:
						handCluster=np.concatenate((handCluster,hand),axis=0)
				if handCluster is not None:
					other.append(handCluster)
					saveFigure(other,pltTitle='Image-'+str(count),clustered=True,saveDir='temp',pltType='3d',axis='xy')
				continue
					
			if count ==34:
				logger.setLevel(logging.DEBUG)
			else:
				logger.setLevel(logging.INFO)
				'''
			chosenClusters,chosenCentroids,chosenDists=getClusterVoting(copy.deepcopy(distClusters),copy.deepcopy(distCentroids),distances,delta,count)
			existingClusters=getExistingClusters(copy.deepcopy(distClusters),copy.deepcopy(distCentroids),currCentroids,currClusters,chosenCentroids,chosenClusters,delta,count)
			chosenClusters=chosenClusters+existingClusters
			chosenPoints=[len(cluster) for cluster in chosenClusters]
			currCentroids=[np.mean(np.array(clust),axis=0) for clust in chosenClusters]
			currClusters=chosenClusters
			logger.info("Time-stamp:%s, Chosen:%s, existing:%s",count,len(chosenClusters),len(existingClusters))
			saveFigure(chosenClusters,pltTitle='Image-'+str(count),clustered=True,saveDir='temp',pltType='2d',axis='xy')
			chosenClusters=[','.join([str(round(x,4)) for x in np.mean(clust,axis=0).tolist()]) for clust in chosenClusters]
			f.write(','.join(chosenClusters)+",TimeStep-"+str(count)+",Points:"+','.join([str(x) for x in chosenPoints])+'\n')
			count+=1
			data_x=[]
			data_y=[]
			data_z=[]
			currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[0] >= 0.2 and line[1] <= radius and line[1] >=-radius and line[2]<=radius and line[2]>=-radius:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
	return	
					
	
def cloudPointParticle(inFile):
	radius=0.6
	N=100
	dt=0.05
	measurementNoise=0.05
	inProximity=0.1
	clusterDist=0.15
	inFile=open(inFile,'r')
	lines=inFile.readlines()
	lines=[line.strip().split(' ') for line in lines]
	data_x=[]
	data_y=[]
	data_z=[]
	direction=[1,-1]
	currTime=None
	count=0
	particle_noise=0.005
	bParticles=None
	prevMean=None
	bCentroids=[]
	bClusters=[]
	hCentroids=[]
	hClusters=[]
	predicts=[]
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			x=np.array([[x,y,z] for x,y,z in zip(data_x,data_y,data_z)])
			x=x.reshape(-1,3)
			bCluster,bMean=getBodyCluster(x,clusterDist,center)	
			hCluster,hMean=getHandClusters(x,0.15,bMean,hClusters)
			if bMean is not None:
				bCentroids.append(bMean)
				bClusters.append(bCluster)
			if len(bCentroids) ==3:
				bMean=np.mean(np.array(bCentroids),axis=0)
				bCluster=np.concatenate((bClusters[0],bClusters[1],bClusters[2]),axis=0)
				if bParticles is None: 
					points=getUniformPoints(bMean,radius,N)
					particles=[]
					for i in range(points.shape[0]):
						particles.append(particle(points[i][0],points[i][1],points[i][2],particle_noise))
					bParticles=Particles(particles)
				bCentroids.pop(0)
				bClusters.pop(0)
				if prevMean is None:
					particles.predict([dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)]])
					prevMean=bMean
				else: 
					dt=bMean-prevMean
					particles.predict(dt.tolist())
					prevMean=bMean
			#particles.plot('Predict')
			cluster=bCluster
			if cluster is not None:
				plt.clf()
				fig=plt.figure()
				ax=fig.add_subplot(111)
				ax.set_xlabel('x')
				ax.set_ylabel('y')
				ax.set_xlim(0,1.5)
				ax.set_ylim(-1,1)
				ax.scatter(cluster[:,0],cluster[:,1],c='r',marker='o')
				plt.savefig('./temp-measure/'+str(count)+'.jpeg')
				plt.close()
				particles.update(bCluster.tolist(),measurementNoise)
				particles.normWeights()
				particles.resampleParticles()
			#particles.plot('Time-step:'+str(count))
			particles.save('Time-step:'+str(count),'./temp','xy')
			count+=1
			data_x=[]
			data_y=[]
			data_z=[]
			currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[0] >= 0.3 and line[1] <= radius and line[1] >=-radius and line[2]<=radius and line[2]>=-radius:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
	return	
					
def plotMultipleDists(distances,target):
	processes=[]
	count=0
	for dist in distances:
		p=mp.Process(target=target,args=('test_12_06/riley_push0d_02_12_06_2019.txt',0.05,0,dist,'Gesture-'+str(dist),'2d','cluster','xy',))
		if count > 4:
			for k in processes:
				k.join()
			count=0
			processes=[]
		processes.append(p)
		count+=1
		p.start()

def KinectPlot(inputFile):
	f=f.readlines()
	indices=[i for i,j in enumerate(f) if "start" in j or "end" in j]
	f=f[indices[0]+1:indices[1]]

saveDir='/home/gmuadmin/Desktop/impactProject/20inches_out'
inDir='/home/gmuadmin/Desktop/impactProject/20inches'
#cloudPointVisualize('exp_19_06/fred_Dshort_headS_01_exp.txt',0.2,0,'Gesture',saveDir=None)
distances=[.1,.15,.2,.25,.3]
plotMultipleDists(distances,cloudPointCluster)
#chooseClusters('camera_test/riley_camera_02_test_exp.txt',distances)

#getConvexHull(None)
#processFiles(inDir,saveDir)
#cloudPointParticle('test_12_06/riley_push0d_02_12_06_2019.txt')
