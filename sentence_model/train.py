from utility import createTrainTest,multiViewDataset,createLogger,trainDataset,multiViewDatasetConcat
from torch.utils.data import DataLoader
from model import sentNet,sentNetClustered,tempNet
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
outFile='train_test_all'
classes=['blank','angry','you','how','weather']
#,'book','that','they','visiting','students']
#classes=['blank','angry','you','how','weather','that','me','wakeup','worried','piano']
targets=[['angry','you'],['weather','how'],['how','you']]
#,['that','weather'],['they','angry','they','visiting'],['they','visiting','students']]
#targets=[['angry','you'],['weather','how'],['how','you'],['that','weather'],['me','wakeup'],['me','worried'],['piano','me'],['me','angry'],['you','wakeup']]
nonManuals=['forward','backward','side','assertion','negation','manual']
#createTrainTest(userDirs,users,testCount,outFile,classes,targets,nonManuals)
#filePath='/home/psanthal/impactProject/impactProject/nn_models/train_test_all'
#dirPath='/scratch/psanthal/impactProject/lowres'
filePath='/home/psanthal/impactProject/sentence_model/train_test_all'
dirPath='/scratch/psanthal/impactProject/clustered_sentences'
#trainDataset=trainDataset(classes)
trainDataset=multiViewDatasetConcat(dirPath,classes,filePath,nonManuals,True,False)
logger=createLogger('/scratch/psanthal','clustered-sent-all-3')
logger.info("Training set total number of samples:%s,xy:%s,yz:%s,xz:%s",len(trainDataset),len(trainDataset.data['xy']),len(trainDataset.data['yz']),len(trainDataset.data['xz']))
saveDir='/scratch/psanthal/clustered-sent-3/'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=sentNetClustered(2048,len(classes),2,10,0.65,False,60)
#modelPath='/scratch/psanthal/clustered-sent-9/model-180.pth'
#net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
optimizer=optim.Adam(net.parameters(),lr=0.00001)
multiViewDataLoader=DataLoader(trainDataset,8,shuffle=True)
criterion=nn.CTCLoss(reduction='none')
m=nn.Softmax(dim=2)
net.train()
for epoch in range(400):
	running_loss=0
	batchCount=0
	for xy,yz,xz,label,tgtLen in multiViewDataLoader:		
		net.zero_grad()
		y=net({'xy':xy.cuda(),'yz':yz.cuda(),'xz':xz.cuda()},60)
		y1=F.log_softmax(y,dim=2)
		inputLen=torch.full(size=(len(xy),), fill_value=12, dtype=torch.long)
		loss=criterion(y1,label.cuda(),inputLen.cuda(),tgtLen.cuda())
		loss=sum(loss)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
		batchCount+=1
		if batchCount==5:
			logger.info("Loss for epoch:%s is: %s",epoch,(running_loss/(batchCount*10)))
			batchCount=0
			running_loss=0
	if epoch%10==0 and epoch > 0:
		print(m(y))	
		torch.save(net.state_dict(),saveDir+'model-'+str(epoch)+'.pth')

