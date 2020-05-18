from keras.models import  Model
from keras.layers import Input, Softmax, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,Softmax
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
import random
#传统的方式
class lossFuncSoft:
    def __init__(self,w=1):
        self.w=w
    def __call__(self,y0,yout0):
        y1 = 1-y0
        yout1 = 1-yout0
        return -K.mean(self.w*y0*K.log(yout0+1e-8)+y1*K.log(yout1+1e-8),axis=-1)

def hitRate(yin,yout,maxD=10):
    yinPos  = K.argmax(yin ,axis=1)
    youtPos = K.argmax(yout,axis=1)
    d       = K.abs(yinPos - youtPos)
    count   = K.sum(K.sign(d+0.1))
    hitCount= K.sum(K.sign(-d+maxD))
    return hitCount/count

def hitRateNp(yin,yout,maxD=6,K=np):
    yinPos  = yin.argmax( axis=1)
    youtPos = yout.argmax(axis=1)
    d       = K.abs(yinPos - youtPos)
    #print(d)
    print(d.mean(axis=(0,1)))
    count   = K.sum(d>-0.1)
    hitCount= K.sum(d<maxD)
    return hitCount/count

def inAndOutFunc(config):
    inputs  = Input(config.inputSize)
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth)]
    dConvL  = [None for i in range(depth)]
    last    = inputs
    for i in range(depth):
        convL[i] = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',\
            activation = config.activationL[i])(last)
        last     = config.poolL[i](pool_size=config.strideL[i],strides=config.strideL[i],padding='same')(convL[i])
    
    for i in range(depth-1,-1,-1):
        dConvL[i]= Conv2DTranspose(config.featureL[i],kernel_size=config.kernelL[i],strides=config.strideL[i],\
            padding='same',activation=config.activationL[i])(last)
        last     = concatenate([dConvL[i],convL[i]],axis=3)
    outputs = Conv2D(config.outputSize[-1],kernel_size=(4,1),strides=(1,1),padding='same',activation='sigmoid')(last)

def inAndOutFuncNew(config):
    inputs  = Input(config.inputSize)
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth)]
    dConvL  = [None for i in range(depth)]
    last    = inputs
    for i in range(depth):
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',\
            activation = config.activationL[i])(last)
        convL[i] = last
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],strides=(1,1),padding='same',\
            activation = config.activationL[i])(last)
        last     = config.poolL[i](pool_size=config.strideL[i],strides=config.strideL[i],padding='same')(last)
    
    for i in range(depth-1,-1,-1):
        dConvL[i]= Conv2DTranspose(config.featureL[i],kernel_size=config.kernelL[i],strides=config.strideL[i],\
            padding='same',activation=config.activationL[i])(last)
        last     = concatenate([dConvL[i],convL[i]],axis=3)
    outputs = Conv2D(config.outputSize[-1],kernel_size=(4,1),strides=(1,1),padding='same',activation='sigmoid')(last)
    return inputs,outputs


class fcnConfig:
    def __init__(self):
        '''
        self.inputSize  = [512,1,1]
        self.outputSize = [512,1,10]
        self.featureL   = [2**(i+1)+20 for i in range(5)]
        self.strideL    = [(4,1),(4,1),(4,1),(2,1),(2,1)]
        self.kernelL    = [(8,1),(8,1),(8,1),(4,1),(2,1)]
        self.activationL= ['relu','relu','relu','relu','relu']
        self.poolL      = [AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D]
        self.lossFunc   = lossFuncSoft
        '''
        self.inputSize  = [1536,1,4]
        self.outputSize = [1536,1,19]
        self.featureL   = [min(2**(i+1)+40,80) for i in range(9)]
        self.strideL    = [(2,1),(2,1),(2,1),(2,1),(2,1),(2,1),(2,1),(2,1),(3,1)]
        self.kernelL    = [(6,1),(6,1),(6,1),(6,1),(6,1),(6,1),(4,1),(2,1),(3,1)]
        self.activationL= ['relu','relu','relu','relu','relu',\
        'relu','relu','relu','relu']
        self.poolL      = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
        AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
        AveragePooling2D]
        self.lossFunc   = lossFuncSoft(w=2)
        self.inAndOutFunc = inAndOutFuncNew
    def inAndOut(self):
        return self.inAndOutFunc(self)

class model(Model):
    def __init__(self,config=fcnConfig(),metrics=hitRateNp,channelList=[1,2,3,4]):
        config.inputSize[-1]=len(channelList)
        self.genM(config)
        self.config = config
        self.metrics = hitRateNp
        self.channelList = channelList
    def genM(self,config):
        inputs, outputs = config.inAndOut()
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x):
        x = self.inx(x)
        return super().predict(x)
    def fit(self,x,y,batchSize=None):
        super().fit(self.inx(x) ,y,batch_size=batchSize)
    def inx(self,x):
        #return x/x.max(axis=(1,2,3),keepdims=True)
        if x.shape[-1]==4:
            x[:,:,:,:2]/=x[:,:,:,:2].std(axis=(1,2,3),keepdims=True)+1e-12
            x[:,:,:,2:]/=x[:,:,:,2:].std(axis=(1,2,3),keepdims=True)+1e-12
        if x.shape[-1]==1:
            x[:,:,:,:]/=x[:,:,:,:].std(axis=(1,2,3),keepdims=True)+1e-12
        if x.shape[-1]==2:
            x[:,:,:,:]/=x[:,:,:,:].std(axis=(1,2,3),keepdims=True)+1e-12
        if x.shape[-1] > len(self.channelList):
            return x[:,:,:,self.channelList]
        else:
            return x
    def __call__(self,x):
        return super(Model, self).__call__(K.tensor(self.inx(x)))
    def train(self,x,y,N=2000,perN=1000,batchSize=None,xTest='',yTest=''):
        indexL = range(x.shape[0])
        #print(indexL)
        lossMin =100
        count0  = 30
        count   = count0
        for i in range(N):
            iL = random.sample(indexL,perN)
            self.fit(x[iL],y[iL],batchSize=batchSize)
            if i%3==0:
                if len(xTest)>0:
                    loss    = self.evaluate(self.inx(xTest),yTest)
                    if loss >= lossMin:
                        count -= 1
                    if loss < lossMin:
                        count = count0
                        lossMin = loss
                    if count ==0:
                        break
                    metrics = self.metrics(yTest,self.predict(xTest))
                    print('test loss: ',loss,' metrics: ',metrics)
            if i%3==0:
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.9)
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(10)):
        y = self.predict(x)
        f = 1/T
        count = x.shape[1]
        for i in range(len(x)):
            timeL = np.arange(count)*delta
            if len(time0L)>0:
                timeL+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0)
            plt.close()
            plt.subplot(3,1,1)
            plt.title('%s%d'%(outputDir,i))
            legend = ['r s','i s',\
            'r h','i h']
            for j in range(x.shape[-1]):
                plt.plot(timeL,self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0]-j,'rbgk'[j],label=legend[j])
            plt.legend()
            plt.xlim(xlim)
            plt.subplot(3,1,2)
            plt.pcolor(timeL,f,y0[i,:,0,:].transpose())
            plt.plot(timeL[pos.astype(np.int)],f,'r')
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlim)
            plt.subplot(3,1,3)
            plt.pcolor(timeL,f,y[i,:,0,:].transpose())
            plt.plot(timeL[pos0.astype(np.int)],f,'b')
            plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlim)
            plt.savefig('%s%d.jpg'%(outputDir,i),dpi=200)
    def predictRaw(self,x):
        yShape = list(x.shape)
        yShape[-1] = self.config.outputSize[-1]
        y = np.zeros(yShape)
        d = self.config.outputSize[0]
        halfD = int(self.config.outputSize[0]/2)
        iL = list(range(0,x.shape[0]-d,halfD))
        iL.append(x.shape[0]-d)
        for i0 in iL:
            y[:,i0:(i0+d)] = x.predict(x[:,i0:(i0+d)])
        return y
        



