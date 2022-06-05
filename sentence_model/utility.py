import multiprocessing as mp
import glob
from torch.utils.data import Dataset
from string import digits
import numpy as np
import torch
import logging 
from sklearn.metrics import accuracy_score,confusion_matrix
import itertools
from asr_evaluation import __main__
import os
import copy
from subprocess import PIPE, run
import torch.nn as nn
import torch
import torch.functional as F
from torchvision.utils import save_image
from PIL import Image

def doMultiProcessing(inFunc,inDir,split,arguments,noJobs=16):
	processes=[]
	count=0
	inFiles=glob.glob(inDir+'/*')
	for i in range(0,len(inFiles),split):
		p = mp.Process(target=inFunc,args=tuple([inFiles[0:0+split]]+arguments))
		if count > noJobs:
			for k in processes:
				k.join()
				processes = []
				count = 0
		p.start()
		count += 1
	if count > 0:
		for k in processes:
			k.join()
	return

def rmDigit(label):
	rm_dig=str.maketrans('','',digits)
	return label.translate(rm_dig)

def getLower(inString):
	return inString.lower()

def getLabel(inFile,nonManual=None,classes=None,withCount=False,includeMan=False,onlyNonManual=False,wordOnly=False):
	if classes is not None:
		if wordOnly:
			try:
				return [k for k,v in enumerate(classes) if v in inFile][0]
			except:
				return None
		if onlyNonManual:
			if any([label in nonManual for label in inFile.split('/')[-1].split('_')[1].split('-')[:-1]]):
				return nonManual.index(inFile.split('/')[-1].split('_')[1].split('-')[:-1][-1])
			else:
				return nonManual.index('manual')
		if 'teach' in inFile:
			return [classes.index(label) for label in ['you','teachme']] 
		if any([label in nonManual for label in inFile.split('/')[-1].split('_')[1].split('-')[:-1]]):
			labels=inFile.split('/')[-1].split('_')[1].split('-')[:-2]
		else:
			labels=inFile.split('/')[-1].split('_')[1].split('-')[:-1]
		frmCls=None
		if "-i-"  in inFile:
			frmCls=labels.index('i')
		elif "-my-" in inFile:
			frmCls=labels.index('my')
		if frmCls is not None:
			labels[frmCls]='me'
		if any([label not in classes for label in labels]):
			return None
		else:
			return [classes.index(label) for label in labels if label in classes]
		
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1].split('.')[0]
	if wordOnly:
		return name.split('_')[1:-1][0]
	if withCount and includeMan:
		return name.split('_')[1:]
	if any([label in name for label in nonManual]):
		return name.split('_')[1:-2]
	else:
		return name.split('_')[1:-1]

def getTargetPadded(targets,classes):
	maxLen=max([len(target) for target in targets])
	targets=[[classes.index(cls) for cls in target] for target in targets]
	paddedTargets=np.full((len(targets),maxLen),fill_value=(len(classes)-1),dtype='i')
	for e,l in enumerate(targets):
		paddedTargets[e,:len(l)]=l
	return paddedTargets

def getUser(inFile):
	user=inFile.strip().split('/')[-1].split('_')[0]
	name=inFile.strip().split('/')[-1]
	return user

def getDedup(k):
	k.sort()
	return list(k for k,_ in itertools.groupby(k))

def createTrainTest(userDirs,users,testCount,outFile,classes,targets=None,nonManual=None):
	classCount={}
	out=open(outFile,'w')
	for user in users:
		if targets is not None:
			for cls in targets:
				classCount[user+'-'+'-'.join(cls)]=0
		else:
			for cls in classes:
				classCount[user+'-'+'-'.join([cls])]=0
	for userDir in userDirs:
		inFiles=glob.glob(userDir+'/*')
		for inFile in inFiles:
			user=getLower(getUser(inFile))
			if targets is not None:
				labels=getLabel(inFile,nonManual)
				fromIndex=None
				if 'me' in classes:
					if any([label not in classes for label in labels if label != 'my' and label != 'i']):
						continue
				else:
					if any([label not in classes for label in labels]):
						continue
				if 'i' in labels:
					fromIndex=labels.index('i')
				elif 'my' in labels:
					fromIndex=labels.index('my')
				if fromIndex is not None:
					labels[fromIndex]='me'
				try:
					count=classCount[user+'-'+'-'.join(labels)]
				except:
					print(inFile,labels)
				if 'teach' in inFile:
					labels=['you','teachme']
				if classCount[user+'-'+'-'.join(labels)] < testCount:
					out.write("Test , {}\n".format(inFile.split('/')[-1]))
					classCount[user+'-'+'-'.join(labels)]+=1
				else:
					out.write("Train , {}\n".format(inFile.split('/')[-1]))
			else:	
				label=getLabel(inFile,wordOnly=True)
				user=getLower(inFile.strip().split('/')[-1].split('_')[0])
				name=inFile.strip().split('/')[-1].split('.')[0]
				dirName=name.split('_')[1]+name.split('_')[2].split('.')[0]
				if classCount[user+'-'+label] < testCount:
					out.write("Test , {}\n".format(user+'_'+dirName))
					classCount[user+'-'+label]+=1
				else:
					out.write("Train , {}\n".format(user+'_'+dirName))

class multiViewDataset(Dataset):
	def __init__(self,dirPath,classes,filePath,nonManual,train=True,bodySeparate=False,frameCount=60,logger=None):
		self.dirPath=dirPath
		self.classes=classes
		self.fileList=[]
		self.trainOnly=train
		self.data=None
		self.labels=[]
		self.frameCount=frameCount
		self.views=['xy','yz','xz']
		self.bodyParts=['body','left','right']
		self.logger=logger
		self.bodySeparate=bodySeparate
		self.nonManual=nonManual
		f=open(filePath,'r')
		f=f.readlines()
		if train:
			f=[f.strip().split(',')[1] for f in f if 'Train' in f]
		else:
			f=[f.strip().split(',')[1] for f in f if 'Test' in f]
		f=[getUser(f)+'_'+'-'.join(getLabel(f,withCount=True,includeMan=True)) for f in f]
		self.fileList=f
		self.fileListLow=[f.lower() for f in f]
		inDirs=glob.glob(dirPath+'/*')
		inDirs=[inDir for inDir in inDirs if inDir.split('/')[-1] in self.fileList or inDir.split('/')[-1] in self.fileListLow]
		self.data,self.labels,self.inFiles=self.loadData(inDirs)
		self.unique_labels=getDedup(copy.deepcopy(self.labels))
		print([[classes[label]  for label in label] for label in self.unique_labels])	
		self.labels=getTargetPadded(self.labels,[i for i in range(len(self.classes))])
		self.tgtLen=[len(label) for label in self.labels]

	#	self.unique_labels=getDedup(self.labels)
	#	print([[classes[label]  for label in label] for label in self.unique_labels])	





	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		if self.trainOnly:
			return torch.tensor(self.data['xy'][idx],dtype=torch.float32),torch.tensor(self.data['yz'][idx],dtype=torch.float32),\
		torch.tensor(self.data['xz'][idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
		torch.tensor(self.tgtLen[idx],dtype=torch.long)
		else:
			return torch.tensor(self.data['xy'][idx],dtype=torch.float32),torch.tensor(self.data['yz'][idx],dtype=torch.float32),\
		torch.tensor(self.data['xz'][idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
		torch.tensor(self.tgtLen[idx],dtype=torch.long),self.inFiles[idx]

	
	def loadData(self,inDirs):
		data={'xy':[],'yz':[],'xz':[]}
		labels=[]
		files=[]
		for inDir in inDirs:
			npFiles=glob.glob(inDir+'/*')
			npFiles=sorted(npFiles)
			data_set=False
	#		if 'jesse' not in inDir:
	#			continue
			if self.bodySeparate:
				for i in range(0,len(npFiles),3):
					view=npFiles[i].split('/')[-1].split('-')[0]
					body=np.load(npFiles[i])
					left=np.load(npFiles[i+1])
					right=np.load(npFiles[i+2])
					if body.shape[0] < self.frameCount:
						if self.logger is not None:
							self.logger.info("For files %s the frame count is %s",npFile,body.shape[0])
						continue
					elif body.shape[0] > self.frameCount:
						start=int(2*(body.shape[0]-self.frameCount)/3)
					else:
						start=0
					data[view].append(np.concatenate((body[start:start+self.frameCount],left[start:start+self.frameCount],right[start:start+self.frameCount]),axis=0))
					data_set=True	
				if data_set:
					labels.append(getLabel(inDir,classes=self.classes))
			else:
				xy=np.load(npFiles[0])
				yz=np.load(npFiles[1])
				xz=np.load(npFiles[2])
				if xy.shape[0] < self.frameCount:
					if self.logger is not None:
						self.logger.info("For files %s the frame count is %s",npFile,xy.shape[0])
					continue
				elif xy.shape[0] > self.frameCount:
					start=int(2*(xy.shape[0]-self.frameCount)/3)
				else:
					start=0
				if getLabel(inDir,nonManual=self.nonManual,classes=self.classes) is None:
					continue
				data['xy'].append(xy[start:start+self.frameCount])
				data['yz'].append(yz[start:start+self.frameCount])
				data['xz'].append(xz[start:start+self.frameCount])
				labels.append(getLabel(inDir,nonManual=self.nonManual,classes=self.classes))
				files.append(inDir)
		return data,labels,files

class multiViewDatasetConcat(Dataset):
	def __init__(self,dirPath,classes,filePath,nonManual=None,train=True,bodySeparate=False,frameCount=60,logger=None,shuffle=False,wordOnly=False):
		self.dirPath=dirPath
		self.classes=classes
		self.fileList=[]
		self.trainOnly=train
		self.data=None
		self.labels=[]
		self.frameCount=frameCount
		self.views=['xy','yz','xz']
		self.bodyParts=['body','left','right']
		self.logger=logger
		self.bodySeparate=bodySeparate
		self.nonManual=nonManual
		self.shuffle=shuffle
		self.wordOnly=wordOnly
		f=open(filePath,'r')
		f=f.readlines()
		if not self.wordOnly:
			if train:
				f=[f.strip().split(',')[1] for f in f if 'Train' in f]
			else:
				f=[f.strip().split(',')[1] for f in f if 'Test' in f]
			f=[getUser(f)+'_'+'-'.join(getLabel(f,withCount=True,includeMan=True)) for f in f]
		else:
			if train:
				f=[f.strip().split(',')[1].strip() for f in f if 'Train' in f]
			else:
				f=[f.strip().split(',')[1].strip() for f in f if 'Test' in f]
		self.fileList=f
		self.fileListLow=[f.lower() for f in f]
		inDirs=glob.glob(dirPath+'/*')
		if self.wordOnly:
			inDirs=[inDir for inDir in inDirs if getLower(inDir.split('/')[-1]) in self.fileList or getLower(inDir.split('/')[-1]) in self.fileListLow]
		else:
			inDirs=[inDir for inDir in inDirs if inDir.split('/')[-1] in self.fileList or inDir.split('/')[-1] in self.fileListLow]
		self.data,self.labels,self.inFiles=self.loadData(inDirs)
		self.unique_labels=getDedup(copy.deepcopy(self.labels))
		if self.wordOnly:
			print([classes[label]  for label in self.unique_labels])	
			print(len(self.data),len(self.labels))
		else:
			print([[classes[label]  for label in label] for label in self.unique_labels])	
		if not self.wordOnly:
			self.tgtLen=[len(label) for label in self.labels]
			self.labels=getTargetPadded(self.labels,[i for i in range(len(self.classes))])
		shape=self.data[0].shape
		for i,x in enumerate(self.data):
			if x.shape!=shape:
				print("Data of different shape found",x.shape,shape,self.inFiles[i])
				exit(0)


	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		if self.wordOnly:
			return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)
		if self.trainOnly:
			return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
			torch.tensor(self.tgtLen[idx],dtype=torch.long)
		else:
			return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
			torch.tensor(self.tgtLen[idx],dtype=torch.long),self.inFiles[idx]

	def loadData(self,inDirs):
		data={'xy':[],'yz':[],'xz':[]}
		labels=[]
		files=[]
		for inDir in inDirs:
			npFiles=glob.glob(inDir+'/*')
			if len(npFiles) == 0:
				continue
			npFiles=sorted(npFiles)
			data_set=False
	#		if 'jesse' not in inDir:
	#			continue
			if self.bodySeparate:
				for i in range(0,len(npFiles),3):
					view=npFiles[i].split('/')[-1].split('-')[0]
					body=np.load(npFiles[i])
					left=np.load(npFiles[i+1])
					right=np.load(npFiles[i+2])
					if body.shape[0] < self.frameCount:
						if self.logger is not None:
							self.logger.info("For files %s the frame count is %s",npFile,body.shape[0])
						continue
					elif body.shape[0] > self.frameCount:
						start=int(2*(body.shape[0]-self.frameCount)/3)
					else:
						start=0
					
					if not self.wordOnly and getLabel(inDir,nonManual=self.nonManual,classes=self.classes) is None:
						continue
					elif getLabel(inDir,classes=self.classes,wordOnly=True) is None:
						continue
					data[view].append(np.concatenate((body[start:start+self.frameCount],left[start:start+self.frameCount],right[start:start+self.frameCount]),axis=0))
					#data[view].append(np.concatenate((body[start:start+self.frameCount],right[start:start+self.frameCount],left[start:start+self.frameCount]),axis=0))
					data_set=True	
				if data_set:
					if self.wordOnly:
						labels.append(getLabel(inDir,classes=self.classes,wordOnly=True))
					#	labels.append(getLabel(inDir,classes=self.classes,wordOnly=True))
					else:
						labels.append(getLabel(inDir,nonManual=self.nonManual,classes=self.classes))
#					labels.append(getLabel(inDir,nonManual=self.nonManual,classes=self.classes))
					#files.append(inDir)
					files.append(inDir)
			else:
				xy=np.load(npFiles[0])
				yz=np.load(npFiles[1])
				xz=np.load(npFiles[2])
				if xy.shape[0] < self.frameCount:
					if self.logger is not None:
						self.logger.info("For files %s the frame count is %s",npFile,xy.shape[0])
					continue
				elif xy.shape[0] > self.frameCount:
					start=int(2*(xy.shape[0]-self.frameCount)/3)
				else:
					start=0
				if not self.wordOnly and getLabel(inDir,nonManual=self.nonManual,classes=self.classes) is None:
					continue
				elif getLabel(inDir,classes=self.classes,wordOnly=True) is None:
					continue
				if len(xy.shape) <4 or len(yz.shape) < 4 or len(xz.shape) < 4:
					continue
				if self.shuffle:
					p=np.random.permutation(15) 
					p+=0
					data['xy'].append(np.concatenate((xy[start:start+self.frameCount][0:0],xy[start:start+self.frameCount][p],xy[start:start+self.frameCount][15:])))
					data['yz'].append(np.concatenate((yz[start:start+self.frameCount][0:0],yz[start:start+self.frameCount][p],yz[start:start+self.frameCount][15:])))
					data['xz'].append(np.concatenate((xz[start:start+self.frameCount][0:0],xz[start:start+self.frameCount][p],xz[start:start+self.frameCount][15:])))
#					data['xz'].append(np.concatenate((xz[start:start+self.frameCount][p],xz[start:start+self.frameCount][10:])))
				else:
					data['xy'].append(xy[start:start+self.frameCount])
					data['yz'].append(yz[start:start+self.frameCount])
					data['xz'].append(xz[start:start+self.frameCount])
				if self.wordOnly:

					labels.append(getLabel(inDir,classes=self.classes,wordOnly=True))
				else:
					labels.append(getLabel(inDir,nonManual=self.nonManual,classes=self.classes))
				files.append(inDir)
		if self.bodySeparate:
			if self.wordOnly:
				data=[np.concatenate((data['xy'][i],data['yz'][i],data['xz'][i]),axis=0).reshape(360,1,100,74) for i in range(len(data['xy']))]
			else:
				data=[np.concatenate((data['xy'][i],data['yz'][i],data['xz'][i]),axis=0).reshape(540,1,100,74) for i in range(len(data['xy']))]

		else:
			data=[np.concatenate((data['xy'][i],data['yz'][i],data['xz'][i]),axis=0) for i in range(len(data['xy']))]
			data=data[:-3]
			labels=labels[:-3]
			files=files[:-3]

		return data,labels,files

class nonManDataset(Dataset):
	def __init__(self,dirPath,classes,filePath,nonManual,train=True,bodySeparate=False,frameCount=60,logger=None):
		self.dirPath=dirPath
		self.classes=classes
		self.fileList=[]
		self.trainOnly=train
		self.data=None
		self.labels=[]
		self.frameCount=frameCount
		self.views=['xy','yz','xz']
		self.bodyParts=['body','left','right']
		self.logger=logger
		self.bodySeparate=bodySeparate
		self.nonManual=nonManual
		f=open(filePath,'r')
		f=f.readlines()
		if train:
			f=[f.strip().split(',')[1] for f in f if 'Train' in f]
		else:
			f=[f.strip().split(',')[1] for f in f if 'Test' in f]
		f=[getUser(f)+'_'+'-'.join(getLabel(f,withCount=True,includeMan=True)) for f in f]
		self.fileList=f
		self.fileListLow=[f.lower() for f in f]
		inDirs=glob.glob(dirPath+'/*')
		inDirs=[inDir for inDir in inDirs if inDir.split('/')[-1] in self.fileList or inDir.split('/')[-1] in self.fileListLow]
		self.data,self.labels,self.inFiles=self.loadData(inDirs)
		shape=self.data[0].shape
		for i,x in enumerate(self.data):
			if x.shape!=shape:
				print("Data of different shape found",x.shape,shape,self.inFiles[i])
				exit(0)
		
		


	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self,idx):
		if self.trainOnly:
			return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long)
		else:
			return torch.tensor(self.data[idx],dtype=torch.float32),torch.tensor(self.labels[idx],dtype=torch.long),\
			self.inFiles[idx]

	def loadData(self,inDirs):
		data={'xy':[],'yz':[],'xz':[]}
		labels=[]
		files=[]
		for inDir in inDirs:
			npFiles=glob.glob(inDir+'/*')
			if len(npFiles) == 0:
				continue
			npFiles=sorted(npFiles)
			data_set=False
	#		if 'jesse' not in inDir:
	#			continue
			if self.bodySeparate:
				for i in range(0,len(npFiles),3):
					view=npFiles[i].split('/')[-1].split('-')[0]
					body=np.load(npFiles[i])
					left=np.load(npFiles[i+1])
					right=np.load(npFiles[i+2])
					if body.shape[0] < self.frameCount:
						if self.logger is not None:
							self.logger.info("For files %s the frame count is %s",npFile,body.shape[0])
						continue
					elif body.shape[0] > self.frameCount:
						start=int(2*(body.shape[0]-self.frameCount)/3)
					else:
						start=0
					data[view].append(np.concatenate((body[start:start+self.frameCount],left[start:start+self.frameCount],right[start:start+self.frameCount]),axis=0))
					data_set=True	
				if data_set:
					labels.append(getLabel(inDir,classes=self.classes))
			else:
				xy=np.load(npFiles[0])
				yz=np.load(npFiles[1])
				xz=np.load(npFiles[2])
				if xy.shape[0] < self.frameCount:
					if self.logger is not None:
						self.logger.info("For files %s the frame count is %s",npFile,xy.shape[0])
					continue
				elif xy.shape[0] > self.frameCount:
					start=int(2*(xy.shape[0]-self.frameCount)/3)
				else:
					start=0
				if getLabel(inDir,nonManual=self.nonManual,classes=self.classes) is None:
					continue
				if len(xy.shape) <4 or len(yz.shape) < 4 or len(xz.shape) < 4:
					continue
				data['xy'].append(xy[start:start+self.frameCount])
				data['yz'].append(yz[start:start+self.frameCount])
				data['xz'].append(xz[start:start+self.frameCount])
				labels.append(getLabel(inDir,nonManual=self.nonManual,classes=self.classes,onlyNonManual=True))
				files.append(inDir)
		data=[np.concatenate((data['xy'][i],data['yz'][i],data['xz'][i]),axis=0) for i in range(len(data['xy']))]

		return data,labels,files

def createLogger(inDir,logFile):
	logging.basicConfig(level=logging.INFO,
		format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
		handlers=[logging.FileHandler("{0}/{1}.log".format(inDir, logFile)),logging.StreamHandler()])
	return logging.getLogger()

def computeAccuracy(labels,predictions,classes):
	return confusion_matrix(labels,predictions,classes),accuracy_score(labels,predictions)

def compute_wer(ref,hyp):
	command="wer "+ref+" "+hyp
	result=run(command,stdout=PIPE,stderr=PIPE,universal_newlines=True,shell=True)
	return result.stdout.strip().split('\n')

def saveConvImages(conv,batch,saveFile):
	x=conv(batch)
	x=x.detach().cpu().numpy()
	
	for i in range(x.shape[0]):
		grid=np.zeros((200,592))
		for j in range(x.shape[1]): 
			if j<8:
				print(((j%8)*74),((j%8)*74)+74)
				grid[0:100,((j%8)*74):((j%8)*74)+74]=x[i,j,:,:].reshape(100,74)
			else:
				grid[100:200,((j%8)*74):((j%8)*74)+74]=x[i,j,:,:].reshape(100,74)
				
		im=Image.fromarray(grid)
		im=im.convert("L")
		im.save(saveFile+'-'+str(i)+'.png')
	
		
