from utility import createTrainTest,multiViewDataset,createLogger,computeAccuracy,multiViewDatasetConcat,compute_wer,nonManDataset,saveConvImages
from torch.utils.data import DataLoader
from model import sentNet,sentNetClustered,tempNet,sentNetClusteredCat,nonManNetClustered
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
from BeamSearch import ctcBeamSearch
from BestPath import ctcBestPath
import os

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
filePath='/home/psanthal/impactProject/sentence_model/man_train_18'
dirPath='/scratch/psanthal/impactProject/clustered_sentences'
modelPath='/scratch/psanthal/nonman-sent-18-2/model-220.pth'
#classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano','want','never','very','book','there','teachme','they','visiting','students']
#targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup'],['me','want','piano'],['that','me','want'],['me','wakeup','never'],['me','very','worried'],['book','there'],['you','teachme'],['they','visiting','students'],['they','angry','they','visiting'],['they','angry','they','worried']]
nonManuals=['forward','backward','side','assertion','negation','manual']
net=nonManNetClustered(2048,len(classes),2,10,0.65,True,60,True)
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
views=['xy','yz','xz']
conv_w={}
conv_b={}
conv={}

for view in views:
	conv_w[view]=net.modules[view].module[0].weight
	conv_b[view]=net.modules[view].module[0].bias
for view in views:
	conv[view]=nn.Sequential(nn.Conv2d(3,16,(5,5),padding=(2,2))).cuda()
	conv[view][0].weight=conv_w[view]
	conv[view][0].bias=conv_b[view]

testDataset=nonManDataset(dirPath,classes,filePath,nonManuals,False,False)
multiViewDataLoader=DataLoader(testDataset,1,shuffle=False)
logger=createLogger('/scratch/psanthal/','nonMan-Eval-log')
logger.info("Testing set total number of samples:%s",len(testDataset))
m=nn.Softmax(dim=1)
predictions=[]
labels=[]
net.eval()
for x,label,inFiles in multiViewDataLoader:		
	for i,view in enumerate(views):
		x1=x[:,i:i+60,:,:,:].reshape(60,3,74,100)
		saveConvImages(conv[view],x1.cuda(),'/scratch/psanthal/nonMan-Images/'+inFiles[0].split('/')[-1]+'-'+view)
	exit(0)
	
	
	
