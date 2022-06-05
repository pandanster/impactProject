from utility import createTrainTest,multiViewDataset,createLogger,computeAccuracy,multiViewDatasetConcat,compute_wer
from torch.utils.data import DataLoader
from model import sentNet,sentNetClustered,sentNetClusteredCat
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
filePath='/home/psanthal/impactProject/sentence_model/train_test_18'
dirPath='/scratch/psanthal/impactProject/highres_sentences'
modelPath='/scratch/psanthal/separated-sent-18-2/model-160.pth'
#classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano','want','never','very','book','there','teachme','they','visiting','students']
#targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup'],['me','want','piano'],['that','me','want'],['me','wakeup','never'],['me','very','worried'],['book','there'],['you','teachme'],['they','visiting','students'],['they','angry','they','visiting'],['they','angry','they','worried']]
nonManuals=['forward','backward','side','assertion','negation','manual']
testDataset=multiViewDatasetConcat(dirPath,classes,filePath,nonManuals,False,True,shuffle=False)
net=sentNet(2048,len(classes),2,10,0.65,True,60)
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
multiViewDataLoader=DataLoader(testDataset,5,shuffle=False)
logger=createLogger('/scratch/psanthal/logFiles/','SentView-Eval-log')
logger.info("Testing set total number of samples:%s",len(testDataset))
m=nn.Softmax(dim=2)
predictions=[]
labels=[]
net.eval()
refFile='/scratch/psanthal/ref'
hypFile='/scratch/psanthal/hyp'
ref=open(refFile,'w')
hyp=open(hypFile,'w')

for x,label,tgtLen,inFile in multiViewDataLoader:		
	o=net(x.cuda(),60)
	o=m(o)
	o=o.detach().cpu().numpy()
	for i in range(o.shape[1]):
		corr=[classes[label] for label in label[i][0:tgtLen[i]]]
		print(inFile[i])
		predicted=','.join([classes[label] for label in np.argmax(o[:,i,:].reshape(8,len(classes)),axis=1).tolist()])
		print(predicted)
		ref.write(' '.join(corr)+'\n')
		pred=ctcBestPath(o[:,i,:].reshape(8,len(classes)),classes)
		hyp.write(pred+'\n')
		#print(ctcBeamSearch(o[:,i,:].reshape(12,len(classes)),classes,None))
ref.flush()
os.fsync(ref.fileno())
hyp.flush()
os.fsync(hyp.fileno())
stdout=compute_wer(refFile,hypFile)
logger.info("The results for Separated body for all with 8 frames with augmentation: %s for sentence count: %s is:",160,18)
logger.info(stdout[0])
logger.info(stdout[1])
logger.info(stdout[2])
logger.info(stdout[3])
