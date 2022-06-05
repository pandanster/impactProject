from utility import createTrainTest,multiViewDataset,createLogger,computeAccuracy,multiViewDatasetConcat,compute_wer,nonManDataset
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
filePath='/home/psanthal/impactProject/sentence_model/unclus_man_train_18'
dirPath='/scratch/psanthal/impactProject/clustered_sentences'
modelPath='/scratch/psanthal/unclus-nonman-18/model-430.pth'
#classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano','want','never','very','book','there','teachme','they','visiting','students']
#targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup'],['me','want','piano'],['that','me','want'],['me','wakeup','never'],['me','very','worried'],['book','there'],['you','teachme'],['they','visiting','students'],['they','angry','they','visiting'],['they','angry','they','worried']]
nonManuals=['forward','backward','side','assertion','negation','manual']
testDataset=nonManDataset(dirPath,classes,filePath,nonManuals,False,False)
net=nonManNetClustered(2048,len(classes),2,10,0.65,True,60,True)
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
multiViewDataLoader=DataLoader(testDataset,8,shuffle=False)
logger=createLogger('/scratch/psanthal/','nonMan-Eval-log')
logger.info("Testing set total number of samples:%s",len(testDataset))
m=nn.Softmax(dim=1)
predictions=[]
labels=[]
net.eval()
for x,label,inFiles in multiViewDataLoader:		
	o=net(x.cuda(),60)
	predictions+=torch.max(m(o),dim=1)[1].cpu().numpy().tolist()
	labels+=label.cpu().numpy().tolist()
confusion,accuracy=computeAccuracy(labels,predictions,[i for i in range(len(nonManuals))])
logger.info("The accuracy for unclus nonMan  model with dropout and highRes: %s is: %s",430,accuracy)
logger.info("The confusion Matrix is")
logger.info(confusion)
