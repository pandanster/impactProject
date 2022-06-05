from utility import createTrainTest,multiViewDataset,createLogger,multiViewDatasetConcat,computeAccuracy
from torch.utils.data import DataLoader
from model import wordNetSep
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
dirPath='/scratch/psanthal/impactProject/highres'
testDataset=multiViewDatasetConcat(dirPath,classes,filePath,train=False,bodySeparate=True,frameCount=40,wordOnly=True)
logger=createLogger('/scratch/psanthal/','logFiles/Word-eval-all')
logger.info("Test set total number of samples:%s",len(testDataset))
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
'''
lowResnet Parameters
hidden_dim,class_size,num_layers,batch_size,dropout,use_cuda,frameCount
'''
net=wordNetSep(2048,len(classes),2,10,0.65,True,40)
modelPath='/scratch/psanthal/separated-word-all-2/model-145.pth'
net.load_state_dict(torch.load(modelPath,map_location='cpu'),strict=False)
m=nn.Softmax(dim=1)
predictions=[]
labels=[]
net.eval()
multiViewDataLoader=DataLoader(testDataset,5,shuffle=False)
for x,label in multiViewDataLoader:		
	o=net(x.cuda(),60)	
	predictions+=torch.max(m(o),dim=1)[1].cpu().numpy().tolist()
	labels+=label.cpu().numpy().tolist()
confusion,accuracy=computeAccuracy(labels,predictions,[i for i in range(len(classes))])
logger.info("The accuracy for separated word  model with dropout and agumentation: %s is: %s",145,accuracy)
logger.info("The confusion Matrix is")
logger.info(confusion)
