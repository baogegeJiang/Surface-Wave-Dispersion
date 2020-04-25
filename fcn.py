from keras.models import  Model
from keras.layers import Input, Softmax, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,Softmax
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
import random
#传统的方式
def lossFuncSoft(y0,yout0,w=1):
    y1 = 1-y0
    yout1 = 1-yout0
    return -K.mean(w*y0*K.log(yout0+1e-8)+y1*K.log(yout1+1e-8),axis=-1)

def hitRate(yin,yout,maxD=10):
    yinPos  = K.argmax(yin ,axis=2)
    youtPos = K.argmax(yout,axis=2)
    d       = K.abs(yinPos - youtPos)
    count   = K.sum(K.sign(d+0.1))
    hitCount= K.sum(K.sign(-d+maxD))
    return hitCount/count

def hitRateNp(yin,yout,maxD=10,K=np):
    yinPos  = K.argmax(yin ,axis=2)
    youtPos = K.argmax(yout,axis=2)
    d       = K.abs(yinPos - youtPos)
    count   = K.sum(d>-0.1)
    hitCount= K.sum(d<maxD)
    return hitCount/count

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
        self.inputSize  = [512,1,1]
        self.outputSize = [512,1,10]
        self.featureL   = [min(2**(i+1)+30,50) for i in range(8)]
        self.strideL    = [(2,1),(2,1),(2,1),(2,1),(2,1),(2,1),(2,1),(2,1)]
        self.kernelL    = [(4,1),(4,1),(4,1),(4,1),(4,1),(4,1),(4,1),(2,1)]
        self.activationL= ['relu','relu','relu','relu','relu',\
        'relu','relu','relu']
        self.poolL      = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
        AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D]
        self.lossFunc   = lossFuncSoft


class model(Model):
    def __init__(self,config=fcnConfig(),metrics=hitRateNp):
        self.genM(config)
        self.config = config
        self.metrics = hitRateNp
    def genM(self,config):
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
        #outputs  = Softmax(axis=3)(last)
        super().__init__(inputs=inputs,outputs=outputs)
        self.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x):
        x = self.inx(x)
        return super().predict(x)
    def fit(self,x,y,batchSize=None):
        super().fit(self.inx(x),y,batch_size=batchSize)
    def inx(self,x):
        return x/x.std(axis=(1,2,3),keepdims=True)
    def __call__(self,x):
        return super(Model, self).__call__(K.tensor(self.inx(x)))
    def train(self,x,y,N=2000,perN=1000,batchSize=None,xTest='',yTest=''):
        indexL = range(x.shape[0])
        #print(indexL)
        lossMin =100
        count0  = 20
        count   =20
        for i in range(N):
            iL = random.sample(indexL,perN)
            self.fit(x[iL],y[iL],batchSize=batchSize)
            if i%10:
                if len(xTest)>0:
                    loss    = self.evaluate(self.inx(xTest),yTest)
                    if loss >= lossMin:
                        count -= 1
                    if loss < lossMin:
                        count = count0
                        lossMin = loss
                    if count ==0:
                        break
                    metrics = self.metrics(xTest,yTest)
                    print('test loss: ',loss,' metrics: ',metrics)
            if i%20:
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
            plt.close()
            plt.subplot(3,1,1)
            plt.plot(timeL,x[i,:,0,0],'b')
            plt.xlim(xlim)
            plt.subplot(3,1,2)
            plt.pcolor(timeL,f,y0[i,:,0,:].transpose())
            plt.gca().semilogy()
            plt.xlim(xlim)
            plt.subplot(3,1,3)
            plt.pcolor(timeL,f,y[i,:,0,:].transpose())
            plt.gca().semilogy()
            plt.xlim(xlim)
            plt.savefig('%s/%d.jpg'%(outputDir,i),dpi=200)
        



