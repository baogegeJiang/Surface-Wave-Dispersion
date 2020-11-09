import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as func
tentype=torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(tentype)
#def lossFunc(y,yout):
#    #yNew=K.concatenate([y,1-y],axis=3)
#    #youtNew=K.concatenate([yout,1-yout],axis=3)
#    #return -K.mean(yNew*K.log(youtNew+1e-9),axis=[0,1,2,3])
#    return -K.mean((y*K.log(yout+1e-9)+(1-y)*(K.log(1-yout+1e-9)))*(y*0+1)*(1+K.sign(y)*wY),axis=[0,1,2,3])
class lossFuncUnet(nn.Module):
    def __init__(self,w0=3,w1=1):
        super(lossFuncUnet, self).__init__()
        self.w0=w0
        self.w1=w1
    def forward(self,y,y0):
        return -torch.mean(y0*torch.log(y+1e-9)*self.w0+\
            (1-y0)*torch.log(1-y+1e-9)*self.w1)

class lossFuncUnetSoft(nn.Module):
    def __init__(self,w0=1,w1=1):
        super(lossFuncUnetSoft, self).__init__()
        self.w0=w0
        self.w1=w1
    def forward(self,y,y0):
        return -torch.mean(y0*torch.log(y+1e-9))

def choiceXY(x,y,n=100):
    n=min(n,x.shape[0])
    indexL=np.arange(x.shape[0])
    indexL=random.sample(indexL.tolist(),n)
    #print(indexL)
    return x[indexL],y[indexL]

def batch(x,batch_size):
    N=x.shape[0]
    #print(N)
    xL=[]
    for i in range(0,N,batch_size):
        xL.append(x[i:min(i+batch_size,N)])
    return xL

class myModule(torch.nn.Module):
    """docstring for """
    def __init__(self):
        super(myModule, self).__init__()
        self.optimizer=[None]
        self.loss=[None]
        self.scheduler = None
    def setLoss(self,lossFunc=lossFuncUnet()):
        self.loss[0]=lossFunc
        return self
    def getLoss(self,x,y0,batch_size=100):
        if not isinstance(x,torch.Tensor):
            xL=batch(x,batch_size)
            y0L=batch(y0,batch_size)
            inx=self.inx(xL[0])
            iny=self.iny(y0L[0])
            loss=self.loss[0](self(inx),iny)
            for i in range(1,len(xL)):
                inx=self.inx(xL[i])
                iny=self.iny(y0L[i])
                print(inx.size(),iny.size())
                loss+=self.loss[0](self(inx),iny)
            return loss
        else:
            y = self(x)
            print(x.size(),y.size(),y0.size)
            return self.loss[0](y,y0)
    def setOptimizer(self,optimizerFunc=torch.optim.Adam,lr=1e-2):
        self.optimizer[0]=optimizerFunc(self.parameters(),lr=lr)
        torch.optim.lr_scheduler.\
        ReduceLROnPlateau(self.optimizer[0], 'min')
        return self.optimizer[0]
    def inx(self,x):
        return x
    def iny(self,y):
        return y
    def outy(self,y):
        return y
    def fit(self,x,y0,batch=40,nb_epoch=3,batch_size=50, verbose=2,isScheduler=False):
        if self.loss[0]==None:
            self.setLoss()
            print('automatic set loss')
        if self.optimizer[0]==None:
            self.setOptimizer()
        N=x.shape[0]
        for i in range(0,N*nb_epoch,batch_size):
            #self.setOptimizer()
            self.optimizer[0].zero_grad()
            tmpX,tmpY0=choiceXY(x,y0,min(batch_size,int(N/2)))
            loss=self.getLoss(tmpX,tmpY0)
            print(loss)
            loss.backward()
            self.optimizer[0].step()
            '''
            for f in self.parameters():
                if isinstance(f,type(None)):
                    continue
                if isinstance(f.data,type(None)):
                    continue
                print(f)
                f.data.sub_(f.grad.data * 1e-5)
            '''
            if self.scheduler!=None and isScheduler:
                self.scheduler.step()
            self.optimizer[0].zero_grad()
        print('trian loss: %f'%self.getLoss(tmpX,tmpY0))
        return self
    def predict(self,x,verbose=0,batch_size=100):
        xL=batch(x,100)
        yL=[self.outy(self(self.inx(tmpX))) for tmpX in xL]
        if isinstance(yL[0],np.ndarray):
            return np.concatenate(yL,axis=0)
        if isinstance(yL[0],torch.Tensor):
            return torch.cat(yL,dim=0)
    def save(self,filename,isAll=False):
        if isAll:
            torch.save(self,filename)
        else:
            torch.save(self.state_dict(),filename)
        return self
    def load_weights(self,filename):
        self.load_state_dict(torch.load('filename').state_dict())
        return self
    def load_weights(self,filename):
        self.load_state_dict(torch.load('filename'))
        return self
    def compile(self,lossFunc=lossFuncUnet(),optimizer='adam',lr=1e-2):
        self.setLoss(lossFunc)
        if optimizer=='adam':
            self.setOptimizer(torch.optim.Adam,lr=lr)
        if optimizer=='SGD':
            self.setOptimizer(torch.optim.SGD,lr=lr)
        return self
    def get_weights(self):
        return self.state_dict()
    def set_weights(self,s0):
        self.load_state_dict(s0)
    def evaluate(self,x=[],y=[]):
        y0=y
        if len(x)>0 and len(y0)>0:
            return self.getLoss(x,y0)
    def summary(self):
        return self

def genModel0(modeltype='norm',phase='p'):
    model=PPP(modeltype,phase)
    model.compile()
    #print(model)
    if modeltype=='norm' and phase=='p':
        #print(1)
        return model,2000,np.arange(1)
    if modeltype=='norm' and phase=='s':
        print(1)
        return model,2000,np.arange(1,2)
    if modeltype=='soft':
        return model,2000,np.arange(3)
    return model,2000,np.arange(1)

class PP(myModule):
    """docstring for ClassName"""
    def __init__(self,modeltype='soft',phase='p'):
        super(PP, self).__init__()
        self.layerN=7
        self.convFilterNL=[25,125,100,125,59,75,100,3]
        self.convKernelNL=[(9,1),(9,1),(5,1),(5,1),(5,1),(5,1),(5,1)]
        self.outKernrlN=(3,1)
        self.convL=nn.ModuleList([nn.Conv2d(self.convFilterNL[i-1],\
            self.convFilterNL[i],\
            self.convKernelNL[i],padding=(int((self.convKernelNL[i][0]-1)/2 ),\
                0))\
            for i in range(self.layerN)])
        self.acL=[F.relu,F.tanh,F.tanh,F.relu,F.tanh,F.relu,F.relu]
        self.poolL=[F.max_pool2d,F.avg_pool2d,F.max_pool2d,F.avg_pool2d,\
        F.max_pool2d,F.max_pool2d,F.max_pool2d]
        #self.poolStrideL = [(5,1),(5,1),(2,1),(2,1),(2,1),(1,1),(1,1)]
        #self.poolKernelNL=[(5,1),(5,1),(2,1),(2,1),(2,1),(1,1),(1,1)]
        self.poolStrideL = [(2,1),(2,1),(2,1),(5,1),(5,1),(1,1),(1,1)]
        self.poolKernelNL=[(3,1),(3,1),(3,1),(6,1),(6,1),(1,1),(1,1)]
        self.dConvL=nn.ModuleList([nn.Conv2d(self.convFilterNL[i]*(1+(i!=self.layerN-1)),\
            self.convFilterNL[i-1],self.convKernelNL[i],stride=(1,1),\
            padding=(int((self.convKernelNL[i][0]-1)/2),0))\
            for i in range(self.layerN)])#ConvTranspose2d
        if modeltype=='soft':
            self.outFilterN=3
            out=nn.Softmax(dim=1)
            self.setLoss(lossFunc=lossFuncUnetSoft)
        if modeltype=='norm':
            self.outFilterN=1
            out=nn.Sigmoid()
        self.outL=nn.ModuleList([nn.Conv2d(self.convFilterNL[-1]*2,self.outFilterN,self.outKernrlN,\
            padding=(int((self.outKernrlN[0]-1)/2 ),0)),out])
        #self.lastL=self.outL[0]
    def inx(self,x):
        return torch.tensor(x).permute(0,3,1,2).float()
    def iny(self,y):
        return torch.tensor(y).permute(0,3,1,2).float()
    def outy(self,y):
        return y.permute(0,2,3,1).cpu().detach().numpy()
    def forward(self,x):
        xL=[None for i in range(self.layerN+1)]
        xL[-1]=x
        for i in range(self.layerN):
            xL[i]=self.poolL[i](self.acL[i](self.convL[i](xL[i-1])),\
                self.poolStrideL[i])
            #print(xL[i].shape)
        for i in range(self.layerN-1,-1,-1):
            xL[i-1]=torch.cat((xL[i-1],F.interpolate(self.acL[i](\
                self.dConvL[i](xL[i])),size=xL[i-1].size()[-2:])),dim=1)
            #print(xL[i-1].shape)
        xout=xL[-1]
        for f in self.outL:
            xout=f(xout)
        return xout

class PPP(PP):
    """docstring for ClassName"""
    def __init__(self,modeltype='soft',phase='p'):
        super(PP, self).__init__()
        self.layerN=7
        self.convFilterNL=[25,125,100,125,59,75,100,3]
        self.convKernelNL=[(9,1),(9,1),(5,1),(5,1),(5,1),(5,1),(5,1)]
        self.outKernrlN=(3,1)
        self.convL1=nn.ModuleList([nn.Conv2d(self.convFilterNL[i-1],\
            self.convFilterNL[i],\
            self.convKernelNL[i],padding=(int((self.convKernelNL[i][0]-1)/2 ),\
                0))\
            for i in range(self.layerN)])
        self.convL2=nn.ModuleList([nn.Conv2d(self.convFilterNL[i],\
            self.convFilterNL[i-(i==self.layerN-1)],\
            self.convKernelNL[i],padding=(int((self.convKernelNL[i][0]-1)/2 ),\
                0))\
            for i in range(self.layerN)])
        self.acL=[F.relu,F.tanh,F.tanh,F.relu,F.tanh,F.relu,F.relu]
        self.pL=[nn.MaxPool2d,nn.AvgPool2d,nn.MaxPool2d,nn.AvgPool2d,\
        nn.MaxPool2d,nn.MaxPool2d,nn.MaxPool2d]
        self.poolL=nn.ModuleList()
        self.poolStrideL = [(2,1),(2,1),(2,1),(5,1),(5,1),(1,1),(1,1)]
        self.poolKernelNL=[(3,1),(3,1),(3,1),(6,1),(6,1),(1,1),(1,1)]
        for i in range(self.layerN):
            self.poolL.append(self.pL[i](self.poolKernelNL[i],self.poolStrideL[i],\
                padding=(int((self.poolKernelNL[i][0]-1)/2),0)))
        self.dConvL1=nn.ModuleList([nn.Conv2d(self.convFilterNL[i]*2,\
            self.convFilterNL[i],self.convKernelNL[i],stride=(1,1),\
            padding=(int((self.convKernelNL[i][0]-1)/2),0))\
            for i in range(self.layerN)])#ConvTranspose2d
        self.dConvL2=nn.ModuleList([nn.Conv2d(self.convFilterNL[i],\
            self.convFilterNL[i-1],self.convKernelNL[i],stride=(1,1),\
            padding=(int((self.convKernelNL[i][0]-1)/2),0))\
            for i in range(self.layerN)])
        if modeltype=='soft':
            self.outFilterN=3
            out=nn.Softmax(dim=1)
            self.setLoss(lossFunc=lossFuncUnetSoft)
        if modeltype=='norm':
            self.outFilterN=1
            out=nn.Sigmoid()
        self.outL=nn.ModuleList([nn.Conv2d(self.convFilterNL[-1],self.outFilterN,self.outKernrlN,\
            padding=(int((self.outKernrlN[0]-1)/2 ),0)),out])
        #self.lastL=self.outL[0]
    def forward(self,x):
        x=self.acL[0](self.convL1[0](x))
        x=self.acL[0](self.convL2[0](x))
        xL=[None for i in range(self.layerN)]
        xL[0]=x
        for i in range(1,self.layerN):
            xL[i]=self.poolL[i](xL[i-1])
            xL[i]=self.acL[i](self.convL1[i](xL[i]))
            xL[i]=self.acL[i](self.convL2[i](xL[i]))
            #print(xL[i].shape)
        for i in range(self.layerN-2,-1,-1):
            xL[i+1]=F.interpolate(xL[i+1],size=xL[i].size()[-2:])
            xL[i]=torch.cat([xL[i+1],xL[i]],dim=1)
            xL[i]=self.acL[i](self.dConvL1[i](xL[i]))
            xL[i]=self.acL[i](self.dConvL2[i](xL[i]))
        xout=xL[0]
        for f in self.outL:
            xout=f(xout)
        return xout

class imagNet(myModule):
    """docstring for ClassName"""
    def __init__(self,modeltype='soft',phase='p'):
        super(imagNet, self).__init__()
        inSize=(64,64)
        self.layerN=4
        self.filerNL=[3*2**i for i in range(self.layerN+1)]
        self.filerNL[-1]=64
        self.kernelSizeL=[(3,3) for i in range(self.layerN)]
        self.poolKernelSizeL=[(3,3) for i in range(self.layerN )]
        self.poolStrideL=[(2,2) for i in range(self.layerN)]
        self.convL=nn.ModuleList([nn.Conv2d(self.filerNL[i],self.filerNL[i+1],\
            self.kernelSizeL[i],padding=(int((self.kernelSizeL[i][0]-1)/2 ),\
                int((self.kernelSizeL[i][1]-1)/2 ))) for i in range(self.layerN)])
        self.acL=[F.relu,F.relu,F.relu,F.relu]
        self.pL=[nn.MaxPool2d,nn.AvgPool2d,nn.MaxPool2d,nn.AvgPool2d]
        self.poolL=nn.ModuleList([self.pL[i](self.poolKernelSizeL[i],self.poolStrideL[i],\
                padding=(int((self.poolKernelSizeL[i][0]-1)/2),\
                    int((self.poolKernelSizeL[i][1]-1)/2))) for i in range(self.layerN)])
        self.fcN=3
        self.fcDL=[4*self.filerNL[-1],32,32,10]
        self.fcL = nn.ModuleList([nn.Linear(self.fcDL[i],self.fcDL[i+1])for i in range(self.fcN)])
        self.out=nn.Softmax(dim=1)
    def xDN(self,x):
        N=1
        for size in x.size()[1:]:
            N*=size
        return N
    def inx(self,x):
        if not isinstance(x,torch.Tensor):
            x=torch.tensor(x).float()
        return x
    def iny(self,y):
        if not isinstance(y,torch.Tensor):
            y=torch.tensor(y).float()
        return y
    def outy(self,y):
        return y
    def forward(self,x):
        for i in range(self.layerN):
            x=self.convL[i](x)
            x=self.acL[i](x)
            x=self.poolL[i](x)
        x = x.view(-1, self.xDN(x))
        for i in range(self.fcN):
            x=self.fcL[i](x)
        return self.out(x)
    def compile(self,lossFunc=lossFuncUnetSoft(),optimizer='adam',lr=1e-3):
        self.setLoss(lossFunc)
        if optimizer=='adam':
            self.setOptimizer(torch.optim.Adam,lr=lr)
        if optimizer=='SGD':
            self.setOptimizer(torch.optim.SGD,lr=lr)
        return self

def loadC10(filename,cn=10):
    with open(filename,'rb') as f:
        d = pickle.load(f, encoding='bytes')
        data=d[b'data'].reshape((-1,3,32,32))
        data=data.astype(np.float)/256
        N=data.shape[0]
        labelM=np.zeros([N,cn])
        labels=d[b'labels']
        for i in range(N):
            labelM[i,labels[i]]=1
        return data,labelM