from utility import createTrainTest,multiViewDataset,createLogger,computeAccuracy,trainDataset,multiViewDatasetConcat,compute_wer
from torch.utils.data import DataLoader
from model import sentNet,sentNetClustered,tempNet,sentNetClusteredCat
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
filePath='/home/psanthal/impactProject/sentence_model/train_test_all'
dirPath='/scratch/psanthal/impactProject/clustered_sentences'
modelPath='/scratch/psanthal/mltTsk-model/model-95.pth'
classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
#classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano']
targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
#targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup']]
nonManuals=['forward','backward','side','assertion','negation','manual']
testDataset=multiViewDataset(dirPath,classes,filePath,nonManuals,False,False)
net=sentNetClustered(2048,len(classes),2,10,0.65,False,60)
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
multiViewDataLoader=DataLoader(testDataset,1,shuffle=False)
logger=createLogger('/scratch/psanthal/','SentView-Eval-log')
logger.info("Testing set total number of samples:%s,xy:%s,yz:%s,xz:%s",len(testDataset),len(testDataset.data['xy']),len(testDataset.data['yz']),len(testDataset.data['xz']))
m=nn.Softmax(dim=2)
predictions=[]
labels=[]
net.eval()
refFile='/scratch/psanthal/ref'
hypFile='/scratch/psanthal/hyp'
ref=open(refFile,'w')
hyp=open(hypFile,'w')

for xy,yz,xz,label,tgtLen,inFile in multiViewDataLoader:		
	print([classes[label] for label in label[0][0:tgtLen[0]]])
	print(label)
	print(inFile)
	continue
	o=net({'xy':xy,'yz':yz,'xz':xz},60)
	#o=net(x,53)
	o=m(o)
	o=o.detach().cpu().numpy()
	for i in range(o.shape[1]):
		corr=[classes[label] for label in label[i][0:tgtLen[i]]]
		print(inFiles[i])
		ref.write(' '.join(corr)+'\n')
		pred=ctcBestPath(o[:,i,:].reshape(12,len(classes)),classes)
		hyp.write(pred+'\n')
		#print(ctcBeamSearch(o[:,i,:].reshape(12,len(classes)),classes,None))
ref.flush()
os.fsync(ref.fileno())
hyp.flush()
os.fsync(hyp.fileno())
stdout=compute_wer(refFile,hypFile)
logger.info("The results for clustered model: %s for sentence count: %s is:",320,3)
logger.info(stdout[0])
logger.info(stdout[1])
logger.info(stdout[2])
logger.info(stdout[3])
