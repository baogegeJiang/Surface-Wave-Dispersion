from keras.models import  Model
from keras.layers import Input, Softmax, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,Softmax
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt
#传统的方式
def lossFuncSoft(y0,yout0,w=1):
    y1 = 1-y0
    yout1 = 1-yout0
    return -K.mean(w*y0*K.log(yout0+1e-8)+y1*K.log(yout1+1e-8),axis=-1)


class fcnConfig:
    def __init__(self):
        self.inputSize  = [512,1,1]
        self.outputSize = [512,1,10]
        self.featureL   = [2**(i+1)+20 for i in range(5)]
        self.strideL    = [(4,1),(4,1),(4,1),(2,1),(2,1)]
        self.kernelL    = [(8,1),(8,1),(8,1),(4,1),(2,1)]
        self.activationL= ['relu','relu','relu','relu','relu']
        self.poolL      = [AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D]
        self.lossFunc   = lossFuncSoft 


class model:
    def __init__(self,config=fcnConfig()):
        self.config = config
        self.m = self.genM(config)
    def genM(self,config):
        inputs = Input(config.inputSize)
        depth  =  len(config.featureL)
        convL  = [None for i in range(depth)]
        dConvL  = [None for i in range(depth)]
        last = inputs
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
        model    = Model(inputs=inputs,outputs=outputs)
        model.compile(loss=config.lossFunc, optimizer='Nadam')
        return model
    def predict(self,x):
        x = self.inx(x)
        return self.m.predict(x)
    def fit(self,x,y):
        self.m.fit(self.inx(x),y)
    def inx(self,x):
        return x/x.std(axis=(1,2,3),keepdims=True)
    def __call__(self,x):
        return self.m(K.tensor(self.inx(x)))
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,f=np.arange(10)):
        y = self.predict(x)
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
            plt.xlim(xlim)
            plt.subplot(3,1,3)
            plt.pcolor(timeL,f,y[i,:,0,:].transpose())
            plt.xlim(xlim)
            plt.savefig('%s/%d.jpg'%(outputDir,i),dpi=200)





