from utility import createTrainTest,multiViewDataset,createLogger,multiViewDatasetConcat
from torch.utils.data import DataLoader
from model import sentNet,sentNetClusteredCat
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
userDirs=['../ASL_sentences/ali_33sents','../ASL_sentences/ari_33sents','../ASL_sentences/arisa_33sents',
	'../ASL_sentences/riley_33sents','../ASL_sentences/jesse_33sents']
users=['riley','ari','arisa','jesse','ali']
#users=['riley','ari','arisa','ali']
#users=['jesse']
testCount=5
outFile='train_test_all'
#classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano','want','never','very','book','there','teachme','they','visiting','students']
#targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup'],['me','want','piano'],['that','me','want'],['me','wakeup','never'],['me','very','worried'],['book','there'],['you','teachme'],['they','visiting','students'],['they','angry','they','visiting'],['they','angry','they','worried']]
nonManuals=['forward','backward','side','assertion','negation','manual']
#createTrainTest(userDirs,users,testCount,outFile,classes,targets,nonManuals)
filePath='/home/psanthal/impactProject/sentence_model/train_test_18'
dirPath='/scratch/psanthal/impactProject/highres_sentences'
trainDataset=multiViewDatasetConcat(dirPath,classes,filePath,nonManuals,True,True)
logger=createLogger('/scratch/psanthal','separated-sent-18-2')
logger.info("Training set total number of samples:%s",len(trainDataset))
saveDir='/scratch/psanthal/separated-sent-18-2/'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=sentNet(2048,len(classes),2,10,0.65,True,60)
#modelPath='/scratch/psanthal/separated-sent-18/model-30.pth'
#net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
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
		loss=criterion(y1,label.to('cuda:2'),inputLen.to('cuda:2'),tgtLen.to('cuda:2'))
		loss=sum(loss)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==10:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/(batchCount*8)))
			batchCount=0
			running_loss=0
	if epoch%5==0 and epoch > 0:
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')

