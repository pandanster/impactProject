from utility import createTrainTest,multiViewDataset,createLogger,multiViewDatasetConcat,nonManDataset
from torch.utils.data import DataLoader
from model import sentNet,sentNetClusteredCat,nonManNetClustered
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
userDirs=['../ASL_sentences/ali_33sents','../ASL_sentences/ari_33sents','../ASL_sentences/arisa_33sents',
	'../ASL_sentences/riley_33sents','../ASL_sentences/jesse_33sents']
users=['riley','ari','arisa','jesse','ali']
testCount=5
outFile='man_train_18'
#classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano','want','never','very','book','there','teachme','they','visiting','students']
#targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup'],['me','want','piano'],['that','me','want'],['me','wakeup','never'],['me','very','worried'],['book','there'],['you','teachme'],['they','visiting','students'],['they','angry','they','visiting'],['they','angry','they','worried']]
nonManuals=['forward','backward','side','assertion','negation','manual']
#createTrainTest(userDirs,users,testCount,outFile,classes,targets,nonManuals)
#filePath='/home/psanthal/impactProject/impactProject/nn_models/train_test_all'
#dirPath='/scratch/psanthal/impactProject/lowres'
filePath='/home/psanthal/impactProject/sentence_model/man_train_18'
dirPath='/scratch/psanthal/impactProject/clustered_sentences'
#trainDataset=trainDataset(classes)
trainDataset=nonManDataset(dirPath,classes,filePath,nonManuals,True,False)
logger=createLogger('/scratch/psanthal','nonman-sent-all-18-2')
logger.info("Training set total number of samples:%s",len(trainDataset))
saveDir='/scratch/psanthal/nonman-sent-18-2/'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=nonManNetClustered(2048,len(classes),2,10,0.65,True,60,True)
modelPath='/scratch/psanthal/nonman-sent-18/model-340.pth'
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
optimizer=optim.Adam(net.parameters(),lr=0.00001)
multiViewDataLoader=DataLoader(trainDataset,8,shuffle=True)
criterion=nn.CrossEntropyLoss()
net.train()
for epoch in range(400):
	running_loss=0
	batchCount=0
	for x,label in multiViewDataLoader:		
		net.zero_grad()
		y=net(x.cuda(),60)
		loss=criterion(y,label.cuda())
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==5:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/(batchCount*10)))
			batchCount=0
			running_loss=0
	if epoch%10==0 and epoch > 0:
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch+340)+'.pth')

