import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import glob
import torch.optim as optim
from string import digits
import logging
import logging.handlers

class Net(nn.Module):
	def __init__(self,hidden_dim,class_size,num_layers,batch_size,dropout,T,use_cuda):
		super(Net,self).__init__()
		self.hidden_dim=hidden_dim
		self.use_cuda=use_cuda
		self.conv1=self.localize(nn.Conv2d(3,32,(20,8),padding=(10,4)))
		self.conv2=self.localize(nn.Conv2d(32,64,(10,4),padding=(5,2)))
		self.conv3=self.localize(nn.Conv2d(64,128,(4,2),padding=(2,1)))
		self.lstm=self.localize(nn.LSTM((8*128),hidden_dim))
		self.fc1=self.localize(nn.Linear(self.hidden_dim,512))
		self.drop=self.localize(nn.Dropout(p=dropout))
		self.fc2=self.localize(nn.Linear(512,class_size))
		self.timeSteps=T
		self.batch=batch_size
		self.num_layers=num_layers
		self.class_size=class_size

	def init_hidden(self,batch):
		return (self.localize(torch.zeros(1,batch,self.hidden_dim)),self.localize(torch.zeros(1,batch,self.hidden_dim)))

	def forward(self,x):
		x1=x.reshape(-1,x.shape[3],x.shape[1],x.shape[2])
		x1=self.localize(torch.tensor(x1,dtype=torch.float32))
		x1=F.max_pool2d(F.relu(self.conv1(x1)),(4,2),2)
		x1=F.max_pool2d(F.relu(self.conv2(x1)),(4,2),2)
		x1=F.max_pool2d(F.relu(self.conv3(x1)),(4,2),2)
		print(x1.shape)
		hidden=self.init_hidden(x.shape[0])
		for t in range(self.timeSteps):
			x2=x1[:,:,:,t:t+1]
			x2=x2.reshape(1,x.shape[0],(8*128))
			o,hidden=self.lstm(x2,hidden)
			o=F.relu(self.fc1(o))
			o=self.drop(o)
			o=self.fc2(o)
		return o.reshape(x.shape[0],self.class_size)


	def localize(self,x):
		if self.use_cuda:
			return x.cuda()
		else:
			return x

	def num_flat_features(self,x):
		size=x.size()[1:]
		num_features=1
		for s in size:
			num_features *=s
		return num_features
	
def localize(x,use_cuda):
	if use_cuda:
		return x.cuda()
	else:
		return x

def getData(inDir,classes,seqLen):
        files=glob.glob(inDir+'*')
        print(len(files))
        data=[]
        labels=[]
        rm_dig=str.maketrans('','',digits)
        shape=None
        for file in files:
                try:
                        label=file.strip().split('/')[6]
                        label=label.translate(rm_dig).split('-')[1].split('.')[0]
                        #print(label)
                except:
                        print('failed::'+file)
                        continue
                if label not in classes:
                        continue
                im=Image.open(file)
                im=np.asarray(im)
                if shape is None:
                        shape=im.shape
                if shape[0] != im.shape[0] or shape[1] != im.shape[1]:
                        print(file)
                data.append(im)
                labels.append(classes.index(label))
        return files,np.array(data),np.array(labels)

			
def train(trainDir,hidden_dim,class_size,num_layers,batch_size,dropout,timeSteps,epochs,classes,seqLen,loadNumpy=False,model=None,use_cuda=False,logFile='log3'):
	logging.basicConfig(
    	level=logging.INFO,
    	format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
   	 handlers=[
        	logging.FileHandler("{0}/{1}.log".format('/scratch/psanthal', logFile)),
        	logging.StreamHandler()])
	logger=logging.getLogger()
	if loadNumpy:
		train_data=np.load(trainDir[0])
		train_labels=np.load(trainDir[1])
	else:
		train_files,train_data,train_labels=getData(trainDir,classes,seqLen) #Need to write this one
		#np.save('/scratch/psanthal/prf_data.npy',train_data)
		#np.save('/scratch/psanthal/prf_labels.npy',train_labels)
	net=Net(hidden_dim,class_size,num_layers,batch_size,dropout,timeSteps,use_cuda)
	if model is not None:
		net.load_state_dict(torch.load(model),strict=False)
		model_next=int(model.split('.')[0].split('-')[-1])+1
	else:
		model_next=0
	criterion=nn.CrossEntropyLoss()
	optimizer=optim.Adam(net.parameters(),lr=0.00001)
	running_loss=0
	for epoch in range(epochs):
		for j in range(0,train_data.shape[0],batch_size):
			net.zero_grad()
			if j+batch_size<train_data.shape[0]:
				x=train_data[j:j+batch_size]
				y=train_labels[j:j+batch_size]
			else:
				x=train_data[j:]
				y=train_labels[j:]
			o=net(x)
			y=localize(torch.tensor(y,dtype=torch.long),use_cuda)
			loss=criterion(o,y)
			loss.backward()
			optimizer.step()
			running_loss+=loss.item()
			if j%20 == 0 and j>0:
				logger.info("Current Loss for epoch::"+str(epoch+model_next)+"::"+str(running_loss/20))
				running_loss=0
		if epoch%15 ==0:
			torch.save(net.state_dict(),'/scratch/psanthal/cnnlstm/cnn-lstm-gpu-model-'+str(epoch+model_next)+'.pth')
		
def evalModel(evalDir,hidden_dim,class_size,num_layers,batch_size,dropout,timeSteps,classes,model,use_cuda):
	eval_files,x,y=getData(evalDir,classes,timeSteps)
	net=Net(hidden_dim,class_size,num_layers,batch_size,dropout,timeSteps,use_cuda)
	net.load_state_dict(torch.load(model,map_location='cpu'),strict=False)
	m=nn.Softmax(dim=1)
	o=torch.max(m(net(x)),dim=1)
	y_pred=o[1].numpy()
	acc=sum([x==y for x,y in zip(y.tolist(),y_pred.tolist())])/float(y.shape[0])
	print("accuracy::"+str(acc))

classes=['ac','alarm','bedroom','calendar','camera','cancel','day','dim','door','email','food','goodmorning','heat','house','kitchen','lightbulb','lock','message','movie','order','phone','picture','rain','raise','restauraunt','room','shopping','snow','sunny','temperature','time','traffic','turndown','turnoff','turnon','wakeup','weather','place','direction','schedule','night','list']	
#trainDir=['/scratch/psanthal/prf_data.npy','/scratch/psanthal/prf_labels.npy']
#evalDir=['/scratch/psanthal/eval_data.npy','/scratch/psanthal/eval_labels.npy']
trainDir='/home/psanthal/code-gest/training-data/train-new2/'
#evalDir='/home/psanthal/code-gest/training-data/test-ash/'
train(trainDir,1024,42,2,10,0.5,13,400,classes,12,loadNumpy=False,model='/scratch/psanthal/cnnlstm/cnn-lstm-gpu-model-221.pth',use_cuda=True,logFile='log-CNN-LSTM2')
#evalModel(evalDir,1024,42,2,10,0.5,12,classes,model='/scratch/psanthal/cnnlstm/cnn-lstm-gpu-model-232.pth',use_cuda=False)
	
#def cnn_model_fn(features,labels,mode):
#TF GRAPH

#data=get_data('aiswarya')
	
