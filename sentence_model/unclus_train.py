from utility import createTrainTest,multiViewDataset,createLogger,multiViewDatasetConcat
from torch.utils.data import DataLoader
from model import sentNet,sentNetClusteredCat,tempNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
userDirs=['../ASL_sentences/ali_33sents','../ASL_sentences/ari_33sents','../ASL_sentences/arisa_33sents',
	'../ASL_sentences/riley_33sents']
#,'../ASL_sentences/jesse_33sents']
#users=['riley','ari','arisa','jesse','ali']
users=['riley','ari','arisa','ali']
testCount=0
outFile='unclust_train_four_18'
#classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano','want','never','very','book','there','teachme','they','visiting','students']
#targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup'],['me','want','piano'],['that','me','want'],['me','wakeup','never'],['me','very','worried'],['book','there'],['you','teachme'],['they','visiting','students'],['they','angry','they','visiting'],['they','angry','they','worried']]
nonManuals=['forward','backward','side','assertion','negation','manual']
#createTrainTest(userDirs,users,testCount,outFile,classes,targets,nonManuals)
filePath='/home/psanthal/impactProject/sentence_model/unclust_train_four_18'
dirPath='/scratch/psanthal/impactProject/unclustered_sentences'
trainDataset=multiViewDatasetConcat(dirPath,classes,filePath,nonManuals,True,False)
logger=createLogger('/scratch/psanthal','unclustered-sent-four-18-2')
logger.info("Training set total number of samples:%s",len(trainDataset))
saveDir='/scratch/psanthal/unclustered-sent-four-18-3/'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
views=['xy','yz','xz']
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net1=sentNetClusteredCat(2048,len(classes),2,10,0.65,True,60,True)
modelPath='/scratch/psanthal/unclustered-sent-four-18/model-270.pth'
net1.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
net=sentNetClusteredCat(2048,len(classes),2,10,0.65,True,60,True)
for view in views:
	for i in [0,3,6,9]:
		net.modules[view].module[i].weight=net1.modules[view].module[i].weight
		net.modules[view].module[i].bias=net1.modules[view].module[i].bias
optimizer=optim.Adam(net.parameters(),lr=0.00001)
multiViewDataLoader=DataLoader(trainDataset,8,shuffle=True)
criterion=nn.CTCLoss(reduction='none')
m=nn.Softmax(dim=2)
net.train()
for epoch in range(400):
	running_loss=0
	batchCount=0
	for x,label,tgtLen in multiViewDataLoader:		
		net.zero_grad()
		y=net(x.cuda(),60)
		y1=F.log_softmax(y,dim=2)
		inputLen=torch.full(size=(len(x),), fill_value=8, dtype=torch.long)
		loss=criterion(y1,label.cuda(),inputLen.cuda(),tgtLen.cuda())
		loss=sum(loss)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==5:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/(batchCount*8)))
			batchCount=0
			running_loss=0
	if epoch%10==0 and epoch > 0:
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')

