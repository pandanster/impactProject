from utility import createTrainTest,multiViewDataset,createLogger,multiViewDatasetConcat
from torch.utils.data import DataLoader
from model import wordNet
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
userDirs=['../ari_23words','../arisa_23words',
	'../riley_new23words','../jesse_23words']
users=['ari','arisa','riley','jesse']
testCount=2
outFile='train_test_words_all'
classes=['teach','you','me','piano','want','very','angry','fire','everyone','huddle','how','today','weather','wakeup','grandmother','never','there','actually','have','must','worried','they','visiting','students']
#createTrainTest(userDirs,users,testCount,outFile,classes)
filePath='/home/psanthal/impactProject/sentence_model/train_test_words_all'
dirPath='/scratch/psanthal/impactProject/unclustered_words'
trainDataset=multiViewDatasetConcat(dirPath,classes,filePath,train=True,frameCount=40,wordOnly=True)
logger=createLogger('/scratch/psanthal','unclustered-word-all')
logger.info("Training set total number of samples:%s",len(trainDataset))
saveDir='/scratch/psanthal/unclustered-word-all/'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=wordNet(2048,len(classes),2,10,0.65,True,40,True)
#modelPath='/scratch/psanthal/clustered-sent-ali-18/model-20.pth'
#net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
optimizer=optim.Adam(net.parameters(),lr=0.000001)
multiViewDataLoader=DataLoader(trainDataset,8,shuffle=True)
criterion=nn.CrossEntropyLoss()
net.train()
for epoch in range(400):
	running_loss=0
	batchCount=0
	for x,label in multiViewDataLoader:		
		net.zero_grad()
		y=net(x.cuda(),40)
		loss=criterion(y,label.cuda())
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==5:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/(batchCount*8)))
			batchCount=0
			running_loss=0
	if epoch%5==0 and epoch > 0:
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')

