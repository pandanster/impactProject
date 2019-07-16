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
from sklearn.neighbors import NearestNeighbors
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances
from sklearn.covariance import ledoit_wolf
import random
import copy
import math
import multiprocessing as mp
from queue import Queue
from scipy.spatial import Delaunay
import logging,sys
from scipy.stats import norm
logging.basicConfig(stream=sys.stderr,level=logging.INFO)
logger=logging.getLogger('cloudPoint')
from mpl_toolkits.mplot3d import proj3d
from matplotlib.ticker import NullLocator

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

def getNonOverlapingPoints(cluster1,cluster2,part=None,clusterParts=None,bodyParts=None,checkOnly=False):
	'''
	dela=Delaunay(cluster1)
	test=dela.find_simplex(cluster2)
	'''
	if type(cluster1) is list:
		print(len(cluster1))
		if len(cluster1) == 0:
			return cluster2
		cluster1=np.array(cluster1)
	if type(cluster2) is list:
		cluster2=np.array(cluster2)
	if cluster1.shape[0] == 0:
		return cluster2
	center,radius=rikersBounding(cluster1)
	nonOverlapping=[]
	for i in range(cluster2.shape[0]):
		if not checkOnly:
			if not checkSphere(cluster2[i],center,radius) and all([np.sum(getNearest(clusterParts[part],cluster2[i].reshape(1,-1),5)[0]) < np.sum(getNearest(clusterParts[tempPart],cluster2[i].reshape(1,-1),5)[0]) for tempPart in bodyParts if part!=tempPart and clusterParts[tempPart] is not None and clusterParts[tempPart].shape[0] > 5 and clusterParts[part].shape[0] >5]):
				nonOverlapping.append(cluster2[i])
			elif not checkSphere(cluster2[i],center,radius) and all([np.sum(getNearest(clusterParts[tempPart],cluster2[i].reshape(1,-1),5)[0]) > 1 for tempPart in bodyParts if part!=tempPart and clusterParts[tempPart] is not None and clusterParts[tempPart].shape[0] > 5 and clusterParts[part].shape[0] >5]):
				nonOverlapping.append(cluster2[i])
		else:
			if not checkSphere(cluster2[i],center,radius):
				nonOverlapping.append(cluster2[i])
	return np.array(nonOverlapping)

def getNearest(points,point,count):
	nbrs=NearestNeighbors(n_neighbors=count,metric='euclidean')
	nbrs.fit(points)
	distances,indices=nbrs.kneighbors(point)
	return distances,indices

def getClusterCount(centroids,delta):
	count=0
	if len(centroids) ==1:
		return 1
	for i,centroid in enumerate(centroids):
		dists,indices=getNearest(np.array(centroids[0:i]+centroids[i+1:]),np.array(centroid).reshape(1,-1),1)
		if dists[0] > delta:
			count+=1
	return count

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
	kinectFile='7_1/hand_data/riley_hand_1_bodyData.txt'
	leftHand,rightHand,body=kinectFitData(kinectFile)
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
					plotClusters=[]
					for u,label in enumerate(labels):	
							indices=[i for i,j in enumerate(pointLabels) if j == label]
							x_=x[indices]	
							plotClusters.append(x_)

					if pltType=='3d':
						if count > 26:
							saveFigure(plotClusters,pltTitle='Image-'+str(count),clustered=True,saveDir='temp',pltType='3d',axis='xy')
						getClusters=input("Enter the list of clusters by their colors:")
						try:
							getClusters=eval(getClusters)
							for color in getClusters:
								clusterIndex=colors.index(color)
								tempCluster=plotClusters[clusterIndex]
								centroid=np.mean(tempCluster,axis=0)
								print("For color:",color,"Centroid::",centroid)
								print("Left::",getHandClusterProb(leftHand,tempCluster))
								print("Right::",getHandClusterProb(rightHand,tempCluster))
								print("Body::",getHandClusterProb(body,tempCluster))
								tempCluster=tempCluster.tolist()
								tempDistances=[math.sqrt(getDist([0,0,0],point)) for point in tempCluster]
								minDistPoint=tempCluster[tempDistances.index(min(tempDistances))]
						except Exception as e:
							print(e)
					else:
						if count==0:
							try:
								os.mkdir(saveDir+'/'+str(distance)+axis)
							except:
								pass
						saveDir=saveDir+'/'+str(distance)+axis
						if axis=='xy':
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir,pltType='2d',axis='xy')
						elif axis=='yz':
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir,pltType='2d',axis='yz')
						else:
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir,pltType='2d',axis='xz')
					clusters.pop(0)
					count+=1
					clusters.append(data)
				else:
					clusters.append(data)
				data_x=[]
				data_y=[]
				data_z=[]
				data_inten=[]
				currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[0] >= 0.3 and line[1] <= 1 and line[1] >=-1 and line[2]<=1 and line[2]>=-1:
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
	clusters=[]
	currTime=None
	count=0
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
					if pltType=='3d':
						if count > 0:
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir='temp',pltType='3d',axis='xy')
					else:
						if axis=='xy':
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir,pltType='2d',axis='xy')
						elif axis=='yz':
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir,pltType='2d',axis='yz')
						else:
							saveFigure(x,pltTitle='Image-'+str(count),clustered=False,saveDir=saveDir,pltType='2d',axis='xz')
					count+=1
					clusters.pop(0)
				clusters.append(data)
			data_x=[]
			data_y=[]
			data_z=[]
			data_inten=[]
			currTime=line[-1]
		else:
			if line[0] <= 1.5 and line[0] >= 0.3 and line[1] <= 1 and line[1] >=-1 and line[2]<=1 and line[2]>=-1:
				data_x.append(line[0])
				data_y.append(line[1])
				data_z.append(line[2])
				data_inten.append(line[3])

def processFiles(inDir,saveDir,distances,kinectFile,Multiview=False,azimuths=None,elevations=None):
	inFiles=glob.glob(inDir+'/*')
	planes=['xy','yz','xz']
	for inFile in inFiles:
		name=inFile.strip().split('/')[-1]
		dirName=name.split('_')[1]+name.split('_')[2].split('.')[0]
		try:
			os.mkdir(saveDir+'/'+dirName)
		except Exception as e:
			logger.debug("Exceptions is :%s",e)		
		if Multiview:
			for azimuth in azimuths: 
				for elevation in elevations:
					try:
						os.mkdir(saveDir+'/'+dirName+'/'+str(azimuth)+'-'+str(elevation))
					except Exception as e:
						logger.debug("Exception is:%s",e)	
			chooseClusters(inFile,distances,kinectFile,saveDir+'/'+dirName,Multiview,azimuths,elevations)
		else:
			for plane in planes:
				try:
					os.mkdir(saveDir+'/'+dirName+'/'+plane)
				except Exception as e:
					logger.debug("Exceptions is :%s",e)
			chooseClusters(inFile,distances,kinectFile,saveDir+'/'+dirName)


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

def saveFigure(x,pltTitle='temp',clustered=False,saveDir=None,pltType='3d',axis=None,Multiview=False,elevations=None,azimuths=None):
	if pltType !='3d':
		plt.clf()
	fig=plt.figure()
	colors=['red','green','blue']
	if pltType=='3d':
		fig.suptitle(pltTitle,fontsize=20)
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_xlim(0,2)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		if clustered:
			for u,x_ in enumerate(x):	
				ax.scatter(x_[:,0],x_[:,1],x_[:,2],c=colors[u],marker='o')
		else:
			ax.scatter(x[:,0],x[:,1],x[:,2],c='r',marker='o')
		plt.show()
	elif Multiview is True:
		ax=fig.add_subplot(111,projection='3d')
		ax.set_xlim(0,2)
		ax.set_ylim(-1,1)
		ax.set_zlim(-1,1)
		ax.grid(False)
		ax.set_axis_off()
		plt.margins(0,0,0)
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		plt.ioff()
		plt.axis('off')
		if clustered:			
			if type(x) is list:
					for u,x_ in enumerate(x):	
						ax.scatter(x_[:,0],x_[:,1],x_[:,2],c=colors[u],marker='o')
			elif type(x) is dict:
				if x['body'] is not None:
					try:
						ax.scatter(x['body'][:,0],x['body'][:,1],x['body'][:,2],c='red',marker='o')
					except:
						logger.debug("Shape for body:%s",x['body'].shape)
						exit(0)
				if x['left'] is not None:
					try:
						ax.scatter(x['left'][:,0],x['left'][:,1],x['left'][:,2],c='green',marker='o')
					except:
						logger.debug("Shape for left:%s",x['left'].shape)
						exit(0)
				if x['right'] is not None:
					try:
						ax.scatter(x['right'][:,0],x['right'][:,1],x['right'][:,2],c='blue',marker='o')
					except:
						logger.debug("Shape for right:%s",x['right'].shape)
						exit(0)
		for azimuth in azimuths:
			for elevation in elevations:
				ax.view_init(elev=elevation,azim=azimuth)
				plt.savefig(saveDir+'/'+str(azimuth)+'-'+str(elevation)+'/'+pltTitle+'.jpeg')
		logger.debug("Multiview is set")
	else:
		ax=fig.add_subplot(111)
		if axis=='xy' or axis == None:
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_xlim(0,2)
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
			ax.set_xlim(0,2)
			ax.set_ylim(-1,1)
			i=0
			j=2
		if clustered:
			if type(x) is list:
				for u,x_ in enumerate(x):	
					ax.scatter(x_[:,i],x_[:,j],c=colors[u],marker='o')
			elif type(x) is dict:
				if x['body'] is not None:
					try:
						ax.scatter(x['body'][:,i],x['body'][:,j],c='red',marker='o')
					except:
						logger.debug("Shape for body:%s",x['body'].shape)
						exit(0)
				if x['left'] is not None:
					try:
						ax.scatter(x['left'][:,i],x['left'][:,j],c='green',marker='o')
					except:
						logger.debug("Shape for left:%s",x['left'].shape)
						exit(0)
				if x['right'] is not None:
					try:
						ax.scatter(x['right'][:,i],x['right'][:,j],c='blue',marker='o')
					except:
						logger.debug("Shape for right:%s",x['right'].shape)
						exit(0)
		else:
			ax.scatter(x[:,i],x[:,j],c='r',marker='o')
		if saveDir is None:
			plt.show()
		else:
			ax.set_axis_off()
			plt.margins(0,0)
			ax.xaxis.set_major_locator(NullLocator())
			ax.yaxis.set_major_locator(NullLocator())
			plt.ioff()
			plt.axis('off')
			plt.savefig(saveDir+'/'+pltTitle+'.jpeg')
			plt.close()
					
def getHandClusterProb(hand,point):
	'''
	normal=multivariate_normal(mu,sigma)
	overlaps=[x for x in clusters.tolist() if normal.pdf(x) > 0.7]
	print("Multivariate Gaussian")
	print(sum([normal.cdf(x) for x in clusters.tolist()]))
	return (len(overlaps)/clusters.shape[0])
	'''
	distances,indices=getNearest(hand,point,30)
	#print(hand[indices])
	return math.sqrt(np.sum(distances))

def getClusterPointOverlap(cluster1,cluster2):
	temp2=None
	temp1=None
	if all([len(cluster)==0 for cluster in cluster1]) or all([len(cluster)==0 for cluster in cluster2]):
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
	return overlap[0].shape[0]/cluster1.shape[0]

def getOverlappingPoints(cluster1,cluster2):
	temp2=None
	temp1=None
	if all([len(cluster)==0 for cluster in cluster1]) or all([len(cluster)==0 for cluster in cluster2]):
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
	return cluster1[overlap[0]]

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
	sigma=ledoit_wolf(cluster1)[0]
	overlaps=[x for x in cluster2.tolist() if multivariate_normal.pdf(x,mu,sigma) > 0.8]
#	logger.debug("Overlap is:%s",len(overlaps)/cluster2.shape[0])
	return len(overlaps)/cluster2.shape[0]
		
def getExistingClusters(distClusters,distCentroids,currentCentroids,currClusters,chosenCentroids,chosenClusters,delta,count):
	existingClusters=[]
	for key,distCluster in distClusters.items():
		for i,cluster in enumerate(distCluster):
			logger.debug("Dist::%s,len of clusters:%s,# Points:%s,Distances:%s,Chosen:%s,ChosenCentroid:%s,distCentroid:%s,existing:%s",key,len(distClusters[key]),cluster.shape[0],[math.sqrt(getDist(distCentroids[key][i],cent)) for cent in chosenCentroids],len(chosenCentroids),chosenCentroids,distCentroids[key][i],len(existingClusters))
			if not any([math.sqrt(getDist(distCentroids[key][i],cent)) < delta for cent in chosenCentroids]):
				for centroid in currentCentroids:
					logger.debug("Distance:%s, existing Overlap: %s, chosen Overlap: %s, centroid Dist%s, current cluster overlap%s:",key,getClusterPointOverlap(existingClusters,[cluster]),getClusterPointOverlap(chosenClusters,[cluster]),math.sqrt(getDist(centroid,distCentroids[key][i])),any([getClusterAreaOverlap([cust],[cluster],count) > 0.6 for cust in currClusters]))
					if (math.sqrt(getDist(centroid,distCentroids[key][i])) < delta or any([getClusterAreaOverlap([cust],[cluster],count) > 0.5 for cust in currClusters])) and getClusterPointOverlap(existingClusters,[cluster]) < 0.7 and getClusterPointOverlap(chosenClusters,[cluster]) < 0.7  and  all([getClusterAreaOverlap([cust],[cluster],count) < 0.4 for cust in chosenClusters]) and all([getClusterAreaOverlap([cust],[cluster],count) < 0.4 for cust in existingClusters]):
						logger.debug("Appending for distance:%s with centroids: %s",key,np.mean(cluster,axis=0))
						existingClusters.append(cluster)
	logger.debug("Existing Clusters:%s for count %s",len(existingClusters),count)
	return existingClusters

def getExistingClusters2(cluster,forCluster):
	chosenClusters=[]
	for point in cluster:
		dists,indices=getNearest(forCluster,point.reshape(1,-1),cluster.shape[0])
		newIndices=[]
		for index,dist in zip(indices.tolist()[0],dists.tolist()[0]):
			if dist < 0.2:
				newIndices.append(index)
		chosenClusters.append(forCluster[newIndices])
	overlapping=None
	for i in range(len(chosenClusters)):
			if overlapping is None:
				overlapping=chosenClusters[i]
			elif hasattr(overlapping,"__iter__"):
				overlapping=getOverlappingPoints([chosenClusters[i]],[overlapping])
	return overlapping

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
				if len(tempClustersSorted) > 2 and min(tempClusterPoints) >=5 and getClusterPointOverlap(chosenClusters,tempClustersSorted) < 0.7:
					chosenDists.append(1/sum(tempDistances))
					chosenClusters.append(tempClustersSorted[int(len(tempClusters)/2)])
					chosenCentroids.append(np.mean(np.array(tempClustersSorted[int(len(tempClusters)/2)]),axis=0).tolist())
	return chosenClusters,chosenCentroids,chosenDists			


def getMin(inData,minElem):
	inData=sorted(inData)
	return inData[minElem]

def getClusterVoting2(distClusters,distCentroids,distances,delta,count,kinectFile):
	chosenClusters=[]
	chosenDists=[]
	chosenCentroids=[]
	kinectFiles=glob.glob(kinectFile+'/*')
	primitive_motions={}
	bodyParts=['left','right','body']
	for kinectFile in kinectFiles:
		name='_'.join(kinectFile.strip().split('/')[-1].split('.')[0].split('_')[:-2])
		primitive_motions[name]={}
		leftHand,rightHand,body=kinectFitData(kinectFile)
		primitive_motions[name]['left']=leftHand
		primitive_motions[name]['right']=rightHand
		primitive_motions[name]['body']=body
	primtive_probs={}
	for primitive in primitive_motions.keys():
		primtive_probs[primitive]={'left':[],'right':[],'body':[]}
	supports=[]
	leftProbs=[]
	rightProbs=[]
	bodyProbs=[]
	distIndex=[]
	pointCount=[]
	partSupports=[]
	for i in range(0,len(distances)):
		logger.debug("The distance currently used:%s",distances[i])
		if len(distClusters[distances[i]]) > 0:
			for v in distCentroids[distances[i]]:
				outIndex=distCentroids[distances[i]].index(v)
				tempClusters=[]
				tempDistances=[]
				support=0
				partSupport={'left':0,'right':0,'body':0}
				for j in range(len(distances)):
					if i==j:
						continue
					if len(distClusters[distances[j]]) >0:
						for k in distCentroids[distances[j]]:			
							try:
								inIndex=distCentroids[distances[j]].index(k)
							except:
									print(k)
									print("all")
									print(distCentroids[distances[j]])
									exit(0)
							if math.sqrt(getDist(distCentroids[distances[i]][outIndex],distCentroids[distances[j]][inIndex])) <= 0.2 and getClusterPointOverlap([distClusters[distances[i]][outIndex]],[distClusters[distances[j]][inIndex]]) > 0.7:
								support+=1
							#if count>33:
							#		logger.debug("Point Overlap::%sCentroid Dist::%s",round(getClusterPointOverlap([distClusters[distances[i]][outIndex]],[distClusters[distances[j]][inIndex]]),6),round(math.sqrt(getDist(distCentroids[distances[i]][outIndex],distCentroids[distances[j]][inIndex])),6))								
				support=support/(len(distances)-1)
				for primitive in primitive_motions.keys():
					leftHandProb=getHandClusterProb(primitive_motions[primitive]['left'],np.array(distCentroids[distances[i]][outIndex]).reshape(1,-1))	
					rightHandProb=getHandClusterProb(primitive_motions[primitive]['right'],np.array(distCentroids[distances[i]][outIndex]).reshape(1,-1))
					bodyProb=getHandClusterProb(primitive_motions[primitive]['body'],np.array(distCentroids[distances[i]][outIndex]).reshape(1,-1))
					primtive_probs[primitive]['left'].append(leftHandProb)
					primtive_probs[primitive]['right'].append(rightHandProb)
					primtive_probs[primitive]['body'].append(bodyProb)
					if min([leftHandProb,rightHandProb,bodyProb]) < 3.5:
						partSupport[bodyParts[([leftHandProb,rightHandProb,bodyProb]).index(min([leftHandProb,rightHandProb,bodyProb]))]]+=1
					if count > 10:
						logger.debug("Primtive:%s,Distance:%s,left:%s,right:%s,body:%s,setting support:%s",primitive,distances[i],round(leftHandProb,4),round(rightHandProb,4),round(bodyProb,4),bodyParts[([leftHandProb,rightHandProb,bodyProb]).index(min([leftHandProb,rightHandProb,bodyProb]))])
				partSupports.append(partSupport)
				#logger.debug("Left Support:%s,Right Support:%s, Body Support:%s",partSupport['left'],partSupport['right'],partSupport['body'])
				supports.append(support)
				distIndex.append((distances[i],outIndex))
				if count > 10:
					logger.debug("Support:%s,left Prob:%s,right Prob:%s,Body Prob%s,Distance:%s,Point Count:%s,Count::%s,Pred::%s,Actual::%s",support,round(leftHandProb,4),round(rightHandProb,4),round(bodyProb,4),distances[i],distClusters[distances[i]][outIndex].shape[0],count,getClusterCount(distCentroids[distances[i]],delta),len(distClusters[distances[i]]))
	clusters={'left':None,'right':None,'body':None}
	tempClusters={'left':None,'right':None,'body':None}
	currDist={'left':None,'right':None,'body':None}
	for i in range(len(supports)):
		if supports[i] > 0.1:
			for part in bodyParts:
				logger.debug("For Part:%s,support is:%s,Distance%s,Cluster size:%s",part,partSupports[i][part],distIndex[i][0],distClusters[distIndex[i][0]][distIndex[i][1]].shape[0])					
				if partSupports[i][part] > 0:
					index=distIndex[i]
					if clusters[part] is None:
						if all([(getClusterPointOverlap([clusters[tempPart]],[distClusters[index[0]][index[1]]]) == 0) for tempPart in bodyParts if clusters[tempPart] is not None and tempPart != part]):
							clusters[part]=distClusters[index[0]][index[1]]
							currDist[part]=index[0]
					elif currDist[part]==index[0] and all([(getClusterPointOverlap([clusters[tempPart]],[distClusters[index[0]][index[1]]]) == 0) for tempPart in bodyParts if clusters[tempPart] is not None and tempPart != part]):
						logger.debug("For distance:%s,the cluster shape is:%s",index[0],distClusters[index[0]][index[1]].shape[0])
						clusters[part]=np.concatenate((clusters[part],distClusters[index[0]][index[1]]),axis=0)
					elif distClusters[index[0]][index[1]].shape[0] > clusters[part].shape[0]:
						if all([(getClusterPointOverlap([clusters[tempPart]],[distClusters[index[0]][index[1]]]) == 0) and  (math.sqrt(getDist(np.mean(clusters[tempPart],axis=0),np.mean(distClusters[index[0]][index[1]],axis=0))) < delta) for tempPart in bodyParts if clusters[tempPart] is not None and tempPart != part]): 
							clusters[part]=distClusters[index[0]][index[1]]
							currDist[part]=index[0]							
						else:
							#for part in bodyParts:
							if clusters[part] is not None:
								logger.debug("For distance:%s,for Part:%s,clusterShape:%s,non-verlapping:%s",index[0],part,distClusters[index[0]][index[1]].shape[0],getNonOverlapingPoints(clusters[part],distClusters[index[0]][index[1]],part,clusters,bodyParts).shape[0])
								if getNonOverlapingPoints(clusters[part],distClusters[index[0]][index[1]],part,clusters,bodyParts).shape[0] >0:
									try:										
										clusters[part]=np.concatenate((clusters[part],getNonOverlapingPoints(clusters[part],distClusters[index[0]][index[1]],part,clusters,bodyParts)),axis=0)
									except:
										logger.debug("Received points:%s",getNonOverlapingPoints(clusters[part],distClusters[index[0]][index[1]],part,clusters,bodyParts).shape)
					
					try:	
						logger.debug("Body Cluster:%s,Distance:%s",clusters['body'].shape[0],currDist['body'])
					except:
						logger.debug("Body Cluster:%s,Distance:%s",clusters['body'],currDist['body'])
					try:
						logger.debug("Left Cluster:%s,Distance:%s",clusters['left'].shape[0],currDist['left'])
					except:
						logger.debug("Left Cluster:%s,Distance:%s",clusters['left'],currDist['left'])
					try:
						logger.debug("Right Cluster:%s,Distance:%s",clusters['right'].shape[0],currDist['right'])
					except:
						logger.debug("Right Cluster:%s,Distance:%s",clusters['right'],currDist['right'])
		else:
			for part in bodyParts:
				logger.debug("For Part:%s,support is:%s,Distance%s,Cluster size:%s",part,partSupports[i][part],distIndex[i][0],distClusters[distIndex[i][0]][distIndex[i][1]].shape[0])					
				if partSupports[i][part] > 0:
					index=distIndex[i]
					if tempClusters[part] is None:
						if all([(getClusterPointOverlap([tempClusters[tempPart]],[distClusters[index[0]][index[1]]]) == 0) for tempPart in bodyParts if tempClusters[tempPart] is not None and tempPart != part]):
							tempClusters[part]=distClusters[index[0]][index[1]]
							currDist[part]=index[0]
					elif currDist[part]==index[0] and all([(getClusterPointOverlap([tempClusters[tempPart]],[distClusters[index[0]][index[1]]]) == 0) for tempPart in bodyParts if tempClusters[tempPart] is not None and tempPart != part]):
						logger.debug("For distance:%s,the cluster shape is:%s",index[0],distClusters[index[0]][index[1]].shape[0])
						tempClusters[part]=np.concatenate((tempClusters[part],distClusters[index[0]][index[1]]),axis=0)
					elif distClusters[index[0]][index[1]].shape[0] > tempClusters[part].shape[0]:
						if all([(getClusterPointOverlap([tempClusters[tempPart]],[distClusters[index[0]][index[1]]]) == 0) and  (math.sqrt(getDist(np.mean(tempClusters[tempPart],axis=0),np.mean(distClusters[index[0]][index[1]],axis=0))) < delta) for tempPart in bodyParts if tempClusters[tempPart] is not None and tempPart != part]): 
							tempClusters[part]=distClusters[index[0]][index[1]]
							currDist[part]=index[0]							
						else:
							#for part in bodyParts:
							if tempClusters[part] is not None:
								logger.debug("For distance:%s,for Part:%s,clusterShape:%s,non-verlapping:%s",index[0],part,distClusters[index[0]][index[1]].shape[0],getNonOverlapingPoints(tempClusters[part],distClusters[index[0]][index[1]],part,tempClusters,bodyParts).shape[0])
								if getNonOverlapingPoints(tempClusters[part],distClusters[index[0]][index[1]],part,tempClusters,bodyParts).shape[0] >0:
									try:										
										tempClusters[part]=np.concatenate((tempClusters[part],getNonOverlapingPoints(tempClusters[part],distClusters[index[0]][index[1]],part,tempClusters,bodyParts)),axis=0)
									except:
										logger.debug("Received points:%s",getNonOverlapingPoints(tempClusters[part],distClusters[index[0]][index[1]],part,tempClusters,bodyParts).shape)
					
					try:	
						logger.debug("Body Cluster:%s,Distance:%s",tempClusters['body'].shape[0],currDist['body'])
					except:
						logger.debug("Body Cluster:%s,Distance:%s",tempClusters['body'],currDist['body'])
					try:
						logger.debug("Left Cluster:%s,Distance:%s",tempClusters['left'].shape[0],currDist['left'])
					except:
						logger.debug("Left Cluster:%s,Distance:%s",tempClusters['left'],currDist['left'])
					try:
						logger.debug("Right Cluster:%s,Distance:%s",tempClusters['right'].shape[0],currDist['right'])
					except:
						logger.debug("Right Cluster:%s,Distance:%s",tempClusters['right'],currDist['right'])
	for part in bodyParts:
		if clusters[part] is None and tempClusters[part] is not None:
			clusters[part]=tempClusters[part]
		elif clusters[part] is None:
			logger.debug("Part %s is None",part)
			clusters[part]=[]	
	if getClusterPointOverlap([clusters['right']],[clusters['body']]) > 0.5:
		center1,radius1=rikersBounding(clusters['right'])
		center2,radius2=rikersBounding(clusters['body'])
		if math.sqrt(getDist(center1,center2)) > 0.1:
			clusters['body']=getNonOverlapingPoints(clusters['right'],clusters['body'],checkOnly=True)
		else:	
			logger.debug("Emptying Right Cluster")
			clusters['right']=[]
	if getClusterPointOverlap([clusters['left']],[clusters['body']]) > 0.5:
		center1,radius1=rikersBounding(clusters['left'])
		center2,radius2=rikersBounding(clusters['body'])
		if math.sqrt(getDist(center1,center2)) > 0.1:
			clusters['body']=getNonOverlapingPoints(clusters['left'],clusters['body'],checkOnly=True)
		else:
			logger.debug("Emptying Left Cluster")
			clusters['left']=[]
	'''	
	currBodyDist=None
	currRightHandDist=None
	currLeftHandDist=None
	bodyCluster=None
	rightHandCluster=None
	leftHandCluster=None
	for i in range(len(supports)):
		bodyDist=getMin(bodyProbs,i)
		rightDist=getMin(rightProbs,i)
		leftDist=getMin(leftProbs,i)
		bodyIndex=distIndex[bodyProbs.index(bodyDist)]
		rightIndex=distIndex[rightProbs.index(rightDist)]
		leftIndex=distIndex[leftProbs.index(leftDist)]
		if supports[bodyProbs.index(bodyDist)] > 0.2:
			if currBodyDist is None:
				currBodyDist=bodyDist
			else:
				if abs(currBodyDist-bodyDist) > delta and bodyCluster is None:
					bodyIndex=distIndex[bodyProbs.index(currBodyDist)]
					bodyCluster=distClusters[bodyIndex[0]][bodyIndex[1]]
				elif bodyDist < 2.5:
					currBodyDist=bodyDist
		if supports[rightProbs.index(rightDist)] > 0.2:
			if currRightHandDist is None:
				currRightHandDist=rightDist
			else:
				if abs(currRightHandDist-rightDist) > delta and rightHandCluster is None:
					rightIndex=distIndex[rightProbs.index(currRightHandDist)]
					rightHandCluster=distClusters[rightIndex[0]][rightIndex[1]]
				elif rightDist < 2.5:
					currRightHandDist=rightDist
		if supports[leftProbs.index(leftDist)] > 0.2:
			if currLeftHandDist is None:
				currLeftHandDist=leftDist
			else:
				if abs(currLeftHandDist-leftDist) > delta and leftHandCluster is None:
					leftIndex=distIndex[leftProbs.index(currLeftHandDist)]
					leftHandCluster=distClusters[leftIndex[0]][leftIndex[1]]
				elif leftDist < 2.5:
					currLeftHandDist=leftDist	
		if bodyCluster is not None and leftHandCluster is not None and rightHandCluster is not None:
			break
	if bodyCluster is None:
		logger.debug("Body is None with dist:%s",currBodyDist)
	if rightHandCluster is None:
		logger.debug("Right Hand is None with dist:%s",currRightHandDist)
	if leftHandCluster is None:
		logger.debug("Left Hand is None with dist:%s",currLeftHandDist)	

	if rightHandCluster is None or getClusterPointOverlap([bodyCluster],[rightHandCluster]) > 0.5 or currRightHandDist > 2.5:
		logger.debug("Setting Right Hand cluster Empty with dist :%s\n",currRightHandDist)
		rightHandCluster=[]
	if leftHandCluster is None or getClusterPointOverlap([bodyCluster],[leftHandCluster]) > 0.5 or currLeftHandDist > 2.5:
		logger.debug("Setting Left Hand cluster Empty with dist :%s\n",currLeftHandDist)
		leftHandCluster=[]	
	if bodyCluster is None:
		bodyIndex=distIndex[bodyProbs.index(currBodyDist)]
		bodyCluster=distClusters[bodyIndex[0]][bodyIndex[1]]
	'''
	return np.array(clusters['body']),np.array(clusters['right']),np.array(clusters['left'])

def chooseClusters(inFile,distances,kinectFile,saveDir=None,Multiview=False,azimuths=None,elevations=None):
	inProximity=0.05
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
	procClusters=[]
	currCentroids=[]
	currClusters={'body':None,'left':None,'right':None}
	f=open('choosen-clusters','w')
	bodyParts=['left','right','body']
	axes=['xy','yz','xz']
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) > 0:
				data=np.array([[x,y,z] for x,y,z in zip(data_x,data_y,data_z)])
				if len(procClusters) > 2:
					x=None
					for clust in procClusters:				
						if x is None:
							x=clust
						else:
							x=np.concatenate((x,clust),axis=0)	
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
					body,right,left=getClusterVoting2(copy.deepcopy(distClusters),copy.deepcopy(distCentroids),distances,delta,count,kinectFile)
					'''
					for part in bodyParts:
						if currClusters[part] is not None:
							print(currClusters[part].shape,part)
						if len(body)==0 and part == 'body' and currClusters[part] is not None:
							if hasattr(getExistingClusters2(currClusters[part],x),"__iter__"):
								body=getExistingClusters2(currClusters[part],x)
						if len(left)==0 and part == 'left' and currClusters[part] is not None:
							if hasattr(getExistingClusters2(currClusters[part],x),"__iter__"):
								left=getExistingClusters2(currClusters[part],x)
						if len(right)==0 and part == 'right' and currClusters[part] is not None:
							if hasattr(getExistingClusters2(currClusters[part],x),"__iter__"):
								right=getExistingClusters2(currClusters[part],x)
					'''
					toPlot={'left':None,'right':None,'body':None}
					if len(left) > 0:
						currClusters['left']=left
						toPlot['left']=left
					else:
						currClusters['left']=None
					if len(right) > 0:
						currClusters['right']=right
						toPlot['right']=right
					else:
						currClusters['right']=None
					if len(body) > 0:
						currClusters['body']=body
						toPlot['body']=body
					else:	
						currClusters['body']=None
					logger.debug("Chossen clusters body:%s,left:%s,right:%s",body.shape[0],left.shape[0],right.shape[0])
					if saveDir is not None and not Multiview:
						for axis in axes:
							saveFigure(toPlot,pltTitle='Image-'+str(count),clustered=True,saveDir=saveDir+'/'+axis,pltType='2d',axis=axis)
					elif Multiview:
						saveFigure(toPlot,pltTitle='Image-'+str(count),clustered=True,saveDir=saveDir,pltType='2d',Multiview=True,azimuths=azimuths,elevations=elevations)
						logger.debug("Multiview set")
					else:
						toPlot=[x for x in toPlot.values() if x is not None]
						saveFigure(toPlot,pltTitle='Image-'+str(count),clustered=True,saveDir=None,pltType='3d',axis=None)
					'''
					body=getNonOverlapingPoints(left,body,checkOnly=True)
					body=getNonOverlapingPoints(right,body,checkOnly=True)		
					existingClusters=getExistingClusters(copy.deepcopy(distClusters),copy.deepcopy(distCentroids),currCentroids,currClusters,chosenCentroids,chosenClusters,delta,count)
					chosenClusters=chosenClusters+existingClusters
					
					handClusters=[]
					for chosen in chosenClusters:
						if getHandClusterProb(leftMu,leftSigma,chosen) > 0.7 or getHandClusterProb(rightMu,rightSigma,chosen) > 0.7:
							print("Left:",getHandClusterProb(leftMu,leftSigma,chosen),"Right:",getHandClusterProb(rightMu,rightSigma,chosen))
							handClusters.append(chosen)
					if len(handClusters) > 0:
						for chosen in chosenClusters:
							print("Left:",getHandClusterProb(leftMu,leftSigma,chosen),"Right:",getHandClusterProb(rightMu,rightSigma,chosen))
					chosenPoints=[len(cluster) for cluster in chosenClusters]
					currCentroids=[np.mean(np.array(clust),axis=0) for clust in chosenClusters]
					currClusters=chosenClusters
					logger.info("Time-stamp:%s, Chosen:%s, existing:%s",count,len(chosenClusters),len(existingClusters))
					if len(handClusters)>0:
#						getClusterVoting2(copy.deepcopy(distClusters),copy.deepcopy(distCentroids),distances,delta,count,leftMu,leftSigma,rightMu,rightSigma)
			#			saveFigure(handClusters,pltTitle='Image-'+str(count),clustered=True,saveDir='temp',pltType='3d',axis='xy')
					if count >10:
						saveFigure(chosenClusters,pltTitle='Image-'+str(count),clustered=True,saveDir='temp',pltType='3d',axis='xy')
					chosenClusters=[','.join([str(round(x,4)) for x in np.mean(clust,axis=0).tolist()]) for clust in chosenClusters]
					#f.write(','.join(chosenClusters)+",TimeStep-"+str(count)+",Points:"+','.join([str(x) for x in chosenPoints])+'\n')
					'''
					count+=1
					procClusters.pop(0)
				procClusters.append(data)
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

def checkSphere(point,center,radius):
	return ((center[0]-point[0])**2+(center[1]-point[1])**2+(center[2]-point[2])**2) < (radius)**2

def getMidPoint(p1,p2):
	if type(p1) is list:
		p1=np.array(p1)
	if type(p2) is list:
		p2=np.array(p2)
	return (p1+p2)/2

def get3DSphere(center,radius):
	return None

def rikersBounding(points):
	if type(points) is not list:
		points=points.tolist()
	x=points[random.randint(0,len(points)-1)]
	distx=[math.sqrt(getDist(x,point)) for point in points]
	y=points[distx.index(max(distx))]
	disty=[math.sqrt(getDist(y,point)) for point in points]
	z=points[disty.index(max(disty))]
	center=getMidPoint(y,z)
	radius=math.sqrt(getDist(y,z))/2
	inSphere=[checkSphere(point,center,radius) for point in points]
	count=0
	while not all(inSphere):
		pointIndex=inSphere.index(False)
		radius=float(math.sqrt(getDist(center,points[pointIndex])))
		radius+=.01
		inSphere=[checkSphere(point,center,radius) for point in points]
		if count>2:
			break
		count+=1
	return np.array(center),radius

def createParticles(cluster,NPoints=100,particle_noise=.0005,clusterType=None):
	center,radius=rikersBounding(cluster)
	print(center,radius)
	if clusterType == "body":
		radius=radius*3/4
	elif clusterType == "right" or clusterType == "left":
		radius=radius/3
	points=getUniformPoints(center,radius,NPoints)
	particles=[]
	for i in range(points.shape[0]):
		particles.append(particle(points[i][0],points[i][1],points[i][2],particle_noise))
	return Particles(particles)

def trackClusters(inFile,distances,kinectFile):
	inProximity=0.05
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
	dt=.05
	measurementNoise=0.05
	procClusters=[]
	currCentroids=[]
	currClusters=[]
	direction=[-1,1]
	f=open('choosen-clusters','w')
	filters={'left':None,'right':None,'body':None}
	trackingFor={'bodyCluster':[],'rightCluster':[],'leftCluster':[],'bodyLife':0,'rightLife':0,'leftLife':0}
	for line in lines:
		line=[float(x.split('::')[1]) for x in line]
		if currTime is None:
			currTime=line[-1]
		elif getTimeDiff(int(currTime),int(line[-1])) > inProximity:
			if len(data_x) > 0:
				data=np.array([[x,y,z] for x,y,z in zip(data_x,data_y,data_z)])
				if len(procClusters) > 2:
					x=None
					for clust in procClusters:				
						if x is None:
							x=clust
						else:
							x=np.concatenate((x,clust),axis=0)	
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
					body,right,left=getClusterVoting2(copy.deepcopy(distClusters),copy.deepcopy(distCentroids),distances,delta,count,kinectFile)
					if len(trackingFor['bodyCluster']) > 0:
						trackingFor['bodyLife']+=1
					if len(trackingFor['rightCluster']) >0:
						trackingFor['rightLife']+=1
					if len(trackingFor['leftCluster']) >0:
						trackingFor['leftLife']+=1
					print("Body Life:",trackingFor['bodyLife'],"Left Life:",trackingFor['leftLife'],"Right Life:",trackingFor['rightLife'])
					if len(body)>0:
						if (len(trackingFor['bodyCluster']) > 0 and math.sqrt(getDist(np.mean(trackingFor['bodyCluster'][-1],axis=0),np.mean(body,axis=0))) < delta) or len(trackingFor['bodyCluster'])==0:
							trackingFor['bodyCluster'].append(body)
						if len(trackingFor['bodyCluster'])>2 and trackingFor['bodyLife'] > 2:
							if filters['body'] is None:
								filters['body']=createParticles(body,clusterType="body")
							else:
								filters['body'].predict([dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)]])
								filters['body'].update(body.tolist(),measurementNoise)
								filters['body'].normWeights()
								filters['body'].resampleParticles()
						if trackingFor['bodyLife'] > 5:
							trackingFor['bodyCluster'].pop(0)
						if len(trackingFor['bodyCluster']) == 0:
							trackingFor['bodyLife']=0
					if len(right)>0:
						if (len(trackingFor['rightCluster']) > 0 and math.sqrt(getDist(np.mean(trackingFor['rightCluster'][-1],axis=0),np.mean(right,axis=0))) < delta) or len(trackingFor['rightCluster'])==0:
							trackingFor['rightCluster'].append(right)	
						if len(trackingFor['rightCluster'])>2 and trackingFor['rightLife'] > 2:
							if filters['right'] is None:
								filters['right']=createParticles(right,clusterType="right")
							else:
								filters['right'].predict([dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)]])
								filters['right'].update(body.tolist(),measurementNoise)
								filters['right'].normWeights()
								filters['right'].resampleParticles()
						if trackingFor['rightLife'] > 5:
							trackingFor['rightCluster'].pop(0)
						if len(trackingFor['rightCluster']) == 0:
							trackingFor['rightLife']=0
					if len(left)>0:
						if (len(trackingFor['leftCluster']) > 0 and math.sqrt(getDist(np.mean(trackingFor['leftCluster'][-1],axis=0),np.mean(left,axis=0))) < delta) or len(trackingFor['leftCluster'])==0:
							trackingFor['leftCluster'].append(left)					
							if len(trackingFor['leftCluster'])>2 and trackingFor['leftLife'] > 2:
								if filters['left'] is None:
									filters['left']=createParticles(left,clusterType="left")
								else:
									filters['left'].predict([dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)],dt*direction[random.randint(0,1)]])
									filters['left'].update(body.tolist(),measurementNoise)
									filters['left'].normWeights()
									filters['left'].resampleParticles()
						if trackingFor['leftLife'] > 5:
							trackingFor['leftCluster'].pop(0)
						if len(trackingFor['leftCluster']) == 0:
							trackingFor['leftLife']=0
					toPlot=[]
					for filter in filters.values():
						if filter is not None:
							toPlot.append(filter.getArray())
					saveFigure(toPlot,clustered=True,pltTitle="Image-"+str(count))
					#saveFigure(chosenClusters,pltTitle='Image-'+str(count),clustered=True,saveDir='temp',pltType='3d',axis='xy')
					count+=1
					procClusters.pop(0)
				procClusters.append(data)
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
		p=mp.Process(target=target,args=('20inches/riley_alarm_02_exp_11_06_2019.txt',0.05,0,dist,'Gesture-'+str(dist),'2d','cluster','xy',))
		if count > 4:
			for k in processes:
				k.join()
			count=0
			processes=[]
		processes.append(p)
		count+=1
		p.start()

'''
Kinect Joints
3- Head joint
2- Neck joint
20- Spine shoulder joint
1- Spine Mid joint
0- Spine base joint
4- Shoulder left
5- Elbow left
6- Wrist Left
8 - Shoulder right
9- Elbow right
10 - Wrist right
'''
def kinectFitData(inputFile):
	f=open(inputFile)
	f=f.readlines()
	indices=[i for i,j in enumerate(f) if "start" in j or "end" in j]
	f=f[indices[0]+1:indices[1]]
	f=[f.strip().split(',') for f in f]
	lf=[(np.array(f)[[6,7,21]]).tolist() for f in f]
	rf=[(np.array(f)[[10,11,23]]).tolist() for f in f]
	body=[(np.array(f)[[3,2,20,1,0]]).tolist() for f in f]
	lf=[[y.split(' ')[1:] for y in x] for x in lf]
	rf=[[y.split(' ')[1:] for y in x] for x in rf]
	body=[[y.split(' ')[1:] for y in x] for x in body]
	lf=[[[float(y) for y in x] for x in f] for f in lf]
	rf=[[[float(y) for y in x] for x in f] for f in rf]
	body=[[[float(y) for y in x] for x in f] for f in body]
	lf=[[x[:3] for x in  f] for f in lf]
	rf=[[x[:3] for x in f] for f in rf]
	body=[[x[:3] for x in f] for f in body]
	lf=[[[x[2],x[0],x[1]] for x in y] for y in lf]
	rf=[[[x[2],x[0],x[1]] for x in y] for y in rf]
	body=[[[x[2],x[0],x[1]] for x in y] for y in body]
	lf=np.array(lf)
	rf=np.array(rf)
	body=np.array(body)
	#print(body.shape)
	lf=lf.reshape(lf.shape[0]*lf.shape[1],-1)
	rf=rf.reshape(rf.shape[0]*rf.shape[1],-1)
	body=body.reshape(body.shape[0]*body.shape[1],-1)
	#print(lf.shape,rf.shape,body.shape)
	leftMu=np.mean(lf,axis=0)
	leftSigma=np.cov(lf,rowvar=False)
#	saveFigure([lf,rf,body],clustered=True)
	rightMu=np.mean(rf,axis=0)
	rightSigma=np.cov(rf,rowvar=False)
	return lf,rf,body		
	
inDir='/mnt/d/impactProject/ali_23words'
saveDir='/mnt/d/impactProject/ali_23words_out'
#cloudPointVisualize('ali_23words/ali_huddle_02.txt',0.05,0,'Gesture',saveDir='temp',pltType='3d',axis='yz')
distances=[0.05,0.075,0.1,.125,.15,.175,.2]
dist=.075
elevations=[0,90,180,270,360]
azimuths=[0,45,90,135,180,225,270,315,360]
#plotMultipleDists(distances,cloudPointCluster)
#cloudPointCluster('ali_23words/ali_students_02.txt',0.05,0,dist,'Gesture-'+str(dist),'3d','cluster','xy',)
kinectFile='primitive_motions'
#kinectFitData(kinectFile)
#chooseClusters('ali_23words/ali_have_01.txt',distances,kinectFile)
#trackClusters('7_1/riley_hand_1.txt',distances,kinectFile)
#getConvexHull(None)
processFiles(inDir,saveDir,distances,kinectFile,True,elevations,azimuths)
#cloudPointParticle('test_12_06/riley_push0d_02_12_06_2019.txt')
#rikersBounding(np.random.randn(10,3))