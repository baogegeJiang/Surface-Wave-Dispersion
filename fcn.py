from keras.models import  Model
from keras.layers import Input, Softmax, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,Softmax,\
  Dropout
import numpy as np
from keras import backend as K
from matplotlib import pyplot as plt   
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import random
from mathFunc import findPos
#默认是float32位的网络
def swish(inputs):
    return (K.sigmoid(inputs) * inputs)
    
get_custom_objects().update({'swish': Activation(swish)})
#传统的方式
class lossFuncSoft:
    # 当有标注的时候才计算权重
    # 这样可以保持结构的一致性
    def __init__(self,w=1):
        self.w=w
    def __call__(self,y0,yout0):
        y1 = 1-y0
        yout1 = 1-yout0
        return -K.mean((self.w*y0*K.log(yout0+1e-8)+y1*K.log(yout1+1e-8))*\
            K.max(y0,axis=1, keepdims=True),\
            axis=-1)

class lossFuncSoftBak:
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

def rateNp(yinPos,youtPos,yinMax,youtMax,maxD=0.03,K=np,minP=0.5):
    threshold = yinPos*maxD
    d       = K.abs(yinPos - youtPos)
    count0   = K.sum(yinMax>0.5)
    count1   = K.sum((yinMax>0.5)*(youtMax>minP))
    hitCount= K.sum((d<threshold)*(yinMax>0.5)*(youtMax>minP))
    return hitCount/count0, hitCount/count1
'''
def rightRateNp(yinPos,youtPos,yinMax,youtMax,maxD=0.03,K=np, minP=0.5):
    threshold = yinPos*maxD
    d       = K.abs(yinPos - youtPos)
    #print(d)
    #print(d.mean(axis=(0,1)))
    count   = K.sum((yinMax>0.5)*(youtMax>minP))
    hitCount= K.sum((d<threshold)*(yinMax>0.5)*(youtMax>minP))
    return hitCount/count
'''


def printRes_old(yin, yout):
    #       0.01  0.8  0.36600 0.9996350
    strL   = 'maxD  minP hitRate rightRate old'
    strfmt = '\n%4.2f %3.1f %7.5f %7.5f'
    yinPos  = yin.argmax( axis=1)
    youtPos = yout.argmax(axis=1)
    yinMax = yin.max(axis=1)
    youtMax = yout.max(axis=1)
    for maxD in [0.03,0.02,0.01]:
        for minP in [0.5,0.7,0.8]:
            hitRate,rightRate = rateNp(\
                yinPos,youtPos,yinMax,youtMax,maxD=maxD,minP=minP)
            strL += strfmt%(maxD, minP, hitRate, rightRate)
    print(strL)
    return strL

def printRes(yin, yout):
    #       0.01  0.8  0.36600 0.9996350
    strL   = 'maxD  minP hitRate rightRate'
    strfmt = '\n%4.2f %3.1f %7.5f %7.5f'
    try:
        yinPos, yinMax = findPos(yin)
        youtPos, youtMax = findPos(yout) 
        for maxD in [0.03,0.02,0.01]:
            for minP in [0.5,0.7,0.8]:
                hitRate,rightRate = rateNp(\
                    yinPos,youtPos,yinMax,youtMax,maxD=maxD,minP=minP)
                strL += strfmt%(maxD, minP, hitRate, rightRate)
    except:
        print('cannot find')
    else:
        pass
    print(strL)
    return strL

def inAndOutFunc(config):
    inputs  = Input(config.inputSize)
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth)]
    dConvL  = [None for i in range(depth)]
    last    = inputs
    for i in range(depth):
        convL[i] = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',\
            activation = config.activationL[i])(last)
        last     = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same')(convL[i])
    
    for i in range(depth-1,-1,-1):
        dConvL[i]= Conv2DTranspose(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=config.strideL[i],\
            padding='same',activation=config.activationL[i])(last)
        last     = concatenate([dConvL[i],convL[i]],axis=3)
    outputs = Conv2D(config.outputSize[-1],kernel_size=(4,1),strides=(1,1),\
        padding='same',activation='sigmoid')(last)
    return inputs,outputs

def inAndOutFuncNew(config):
    inputs  = Input(config.inputSize)
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth)]
    dConvL  = [None for i in range(depth)]
    last    = inputs
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',activation = config.activationL[i],\
            name=name+layerStr+'0',kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii])(last)
        convL[i] = last
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',activation = config.activationL[i],\
            name=name+layerStr+'1',kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        last     = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same')(last)
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        layerStr='_%d_'%i
        dConvL[i]= Conv2DTranspose(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=config.strideL[i],padding='same',activation=config.activationL[i],\
            name=name+layerStr+'0',kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        last     = concatenate([dConvL[i],convL[i]],axis=3)
    outputs = Conv2D(config.outputSize[-1],kernel_size=(4,1),strides=(1,1),padding='same',activation='sigmoid',name='dconv_out')(last)
    return inputs,outputs

def inAndOutFuncNewV2(config):
    inputs  = Input(config.inputSize)
    depth   =  len(config.featureL)
    convL   = [None for i in range(depth+1)]
    dConvL  = [None for i in range(depth+1)]
    last    = inputs
    for i in range(depth):
        if i <4:
            name = 'conv'
        else:
            name = 'CONV'
        layerStr='_%d_'%i
        '''
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',activation = config.activationL[i],\
            name=name+layerStr+'0',kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        '''
        '''
        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii])(last)
        '''
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',activation = config.activationL[i],\
            name=name+layerStr+'0',kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        convL[i] =last
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',activation = config.activationL[i],\
            name=name+layerStr+'1',kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)
        last     = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same')(last)
        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii])(last)
    convL[depth] =last
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        
        for j in range(i,-1,-1):
            layerStr='_%d_%d'%(i,j)
            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
            strides=config.strideL[j],padding='same',activation=config.activationL[j],\
            name=name+layerStr+'0',kernel_initializer=config.initializerL[j],\
            bias_initializer=config.bias_initializerL[j])(convL[j+1])
            if j in config.dropOutL:
                jj   = config.dropOutL.index(j)
                dConvL[j] =  Dropout(config.dropOutRateL[jj])(dConvL[j])
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=3)
        
    outputs = Conv2D(config.outputSize[-1],kernel_size=(4,1),strides=(1,1),\
        padding='same',activation='sigmoid',name='dconv_out')(convL[0])
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
        self.inputSize     = [4096*3,1,4]
        self.outputSize    = [4096*3,1,50]
        self.featureL      = [min(2**(i+1)+20,80) for i in range(7)]
        self.featureL      = [30,40,60,60,80,60,40]
        self.featureL      = [15,20,20,25,25,40,60]
        self.featureL      = [50,50,75,75,100,100,125]
        #[min(2**(i+1)+20,60) for i in range(6)]#[min(2**(i+1)+80,120) for i in range(8)]#40
        self.strideL       = [(4,1),(4,1),(4,1),(4,1),(4,1),(4,1),(3,1),\
        (4,1),(2,1),(2,1),(2,1)]
        self.kernelL       = [(8,1),(8,1),(8,1),(8,1),(8,1),(8,1),(3,1),\
        (8,1),(4,1),(4,1),(4,1)]
        self.initializerL  = ['truncated_normal' for i in range(10)]
        self.initializerL  = ['he_normal' for i in range(10)]
        self.bias_initializerL=['random_normal' for i in range(10)]
        self.bias_initializerL=['he_normal' for i in range(10)]
        #self.strideL    = [(4,1),(4,1),(4,1),(4,1),(4,1),(2,1),(4,1),(4,1),(2,1),(2,1),(2,1)]
        #self.kernelL    = [(8,1),(8,1),(8,1),(8,1),(8,1),(4,1),(8,1),(4,1),(4,1),(4,1),(4,1)]
        self.dropOutL     = [1,3,5,7]#[1,3,5,7]
        self.dropOutRateL = [0.1,0.1,0.1,0.1]
        self.activationL  = ['relu','relu','relu','relu','relu',\
        'relu','relu','relu','relu','relu','relu']
        self.activationL  = ['relu','relu']+['swish' for i in range(4)]+['relu']
        self.poolL        = [AveragePooling2D,AveragePooling2D,MaxPooling2D,\
        AveragePooling2D,AveragePooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
        MaxPooling2D,AveragePooling2D,MaxPooling2D]
        self.poolL        = [MaxPooling2D,AveragePooling2D,MaxPooling2D,\
        AveragePooling2D,MaxPooling2D,MaxPooling2D,MaxPooling2D,AveragePooling2D,\
        MaxPooling2D,AveragePooling2D,MaxPooling2D]
        self.lossFunc     = lossFuncSoft(w=10)#10
        self.inAndOutFunc = inAndOutFuncNewV2
    def inAndOut(self):
        return self.inAndOutFunc(self)

class xyt:
    def __init__(self,x,y,t=''):
        self.x = x
        self.y = y
        self.t = t
        self.timeDisKwarg={'sigma':-1}
    def __call__(self,iL):
        if not isinstance(iL,np.ndarray):
            iL= np.array(iL).astype(np.int)
        if len(self.t)>0:
            tout = self.t[iL]
        else:
            tout = self.t
        self.iL = iL
        return self.x[iL],self.y[iL],tout
    def __len__(self):
        return self.x.shape[0]

class model(Model):
    def __init__(self,weightsFile='',config=fcnConfig(),metrics=rateNp,\
        channelList=[1,2,3,4]):
        config.inputSize[-1]=len(channelList)
        self.genM(config)
        self.config = config
        self.Metrics = metrics
        self.channelList = channelList
        if len(weightsFile)>0:
            model.load_weights(weightsFile)
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
        '''
        if x.shape[-1]==4:
            x[:,:,:,:2]/=x[:,:,:,:2].std(axis=(1,2,3),keepdims=True)+1e-12
            x[:,:,:,2:]/=x[:,:,:,2:].std(axis=(1,2,3),keepdims=True)+1e-12
        '''
        timeN0 = x.shape[1]
        timeN  = (x!=0).sum(axis=1,keepdims=True)
        timeN *= 1+0.2*(np.random.rand(*timeN.shape)-0.5)
        x/=(x.std(axis=(1,2),keepdims=True))*(timeN0/timeN)**0.5
        if x.shape[-1] > len(self.channelList):
            return x[:,:,:,self.channelList]
        else:
            return x
    def __call__(self,x):
        return super(Model, self).__call__(K.tensor(self.inx(x)))
    def train(self,x,y,**kwarg):
        if 't' in kwarg:
            t = kwarg['t']
        else:
            t = ''
        XYT = xyt(x,y,t)
        self.trainByXYT(XYT,**kwarg)
    def trainByXYT(self,XYT,N=2000,perN=200,batchSize=None,xTest='',\
        yTest='',k0 = 5e-4,t='',count0=3):
        if k0>1:
            K.set_value(self.optimizer.lr, k0)
        indexL = range(len(XYT))
        #print(indexL)
        lossMin =100
        count   = count0
        w0 = self.get_weights()
        resStr=''
        #print(self.metrics)
        for i in range(N):
            iL = random.sample(indexL,perN)
            x, y , t0L = XYT(iL)
            #print(XYT.iL)
            self.fit(x ,y,batchSize=batchSize)
            if i%10==0:
                if len(xTest)>0:
                    loss    = self.evaluate(self.inx(xTest),yTest)
                    if loss >= lossMin:
                        count -= 1
                    if loss > 3*lossMin:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if loss < lossMin:
                        count = count0
                        lossMin = loss
                        w0 = self.get_weights()
                        print('find better')
                    if count ==0:
                        break
                    #print(self.metrics)
                    resStr+='\n %d loss :%f'%(i,loss)
                    print('test loss: ',loss,'sigma: ',XYT.timeDisKwarg['sigma'],\
                        'w: ',self.config.lossFunc.w)
                    if i%30==0 and i>10:
                        yout = self.predict(xTest)
                        resStr+='\n'+printRes_old(yTest, yout)
                        if i%60==0:
                            printRes(yTest, yout)
            if i%5==0:
                print('learning rate: ',self.optimizer.lr)
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.95)
            if i>10 and i%5==0:
                perN += int(perN*0.05)
                perN = min(1000, perN)
        self.set_weights(w0)
        return resStr
    def trainByXYTCross(self,self1,XYT0,XYT1,N=2000,perN=100,batchSize=None,\
        xTest='',yTest='',k0 = -1,t='',per1=0.5):
        #XYT0 syn
        #XYT1 real
        if k0>1:
            K.set_value(self.optimizer.lr, k0)
        indexL0 = range(len(XYT0))
        indexL1 = range(len(XYT1))
        #print(indexL)
        lossMin =100
        count0  = 10
        count   = count0
        w0 = self.get_weights()
        #print(self.metrics)
        for i in range(N):
            is0 =False
            if (i < 10) or (np.random.rand()<per1  and i <20):
                print('1')
                is0 =False
                XYT = XYT1
                iL = random.sample(indexL1,perN)
                SELF = self1
            else:
                print('0')
                is0 = True
                XYT = XYT0
                iL = random.sample(indexL0,perN)
                SELF = self
            x, y , t0L = XYT(iL)   
            #print(XYT.iL)
            SELF.fit(x ,y ,batchSize=batchSize)
            if  is0:
                self1.set_weights(self.get_weights())
            else:
                self.set_weights(self1.get_weights())
            if i%3==0 and (is0 or per1>=1):
                if len(xTest)>0:
                    loss    = self.evaluate(self.inx(xTest),yTest)
                    lossM = self.Metrics(yTest,self.predict(xTest))
                    if loss >= lossMin:
                        count -= 1
                    if loss > 3*lossMin:
                        self.set_weights(w0)
                        #count = count0
                        print('reset to smallest')
                    if loss < lossMin:
                        count = count0
                        lossMin = loss
                        w0 = self.get_weights()
                    if count ==0:
                        break
                    #print(self.metrics)
                   
                    print('test loss: ',loss,' metrics: ',lossM,'sigma: ',\
                        XYT.timeDisKwarg['sigma'],'w: ',self.config.lossFunc.w)
            if i%5==0:
                print('learning rate: ',self.optimizer.lr)
                K.set_value(self.optimizer.lr, K.get_value(self.optimizer.lr) * 0.9)
                K.set_value(self1.optimizer.lr, K.get_value(self1.optimizer.lr) * 0.9)
            if i==50:
                perN = 100
            if i>10 and i%5==0:
                perN += int(perN*0.02)
        self.set_weights(w0)
    def show(self, x, y0,outputDir='predict/',time0L='',delta=0.5,T=np.arange(19),fileStr=''):
        y = self.predict(x)
        f = 1/T
        count = x.shape[1]
        for i in range(len(x)):
            #print('show',i)
            timeL = np.arange(count)*delta
            if len(time0L)>0:
                timeL+=time0L[i]
            xlim=[timeL[0],timeL[-1]]
            xlimNew=[0,500]
            #xlim=xlimNew
            tmpy0=y0[i,:,0,:]
            pos0  =tmpy0.argmax(axis=0)
            tmpy=y[i,:,0,:]
            pos  =tmpy.argmax(axis=0)
            plt.close()
            plt.figure(figsize=[12,8])
            plt.subplot(4,1,1)
            plt.title('%s%d'%(outputDir,i))
            legend = ['r s','i s',\
            'r h','i h']
            for j in range(x.shape[-1]):
                plt.plot(timeL,self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0]-j,'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            #plt.legend()
            plt.xlim(xlim)
            plt.subplot(4,1,2)
            #plt.clim(0,1)
            plt.pcolor(timeL,f,y0[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            plt.plot(timeL[pos.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,3)
            plt.pcolor(timeL,f,y[i,:,0,:].transpose(),cmap='bwr',vmin=0,vmax=1)
            #plt.clim(0,1)
            plt.plot(timeL[pos0.astype(np.int)],f,'k',linewidth=0.5,alpha=0.5)
            plt.ylabel('f/Hz')
            plt.xlabel('t/s')
            plt.gca().semilogy()
            plt.xlim(xlimNew)
            plt.subplot(4,1,4)
            delta = timeL[1] -timeL[0]
            N = len(timeL)
            fL = np.arange(N)/N*1/delta
            for j in range(x.shape[-1]):
                spec=np.abs(np.fft.fft(self.inx(x[i:i+1,:,0:1,j:j+1])[0,:,0,0])).reshape([-1])
                plt.plot(fL,spec/(spec.max()+1e-16),'rbgk'[j],\
                    label=legend[j],linewidth=0.3)
            plt.xlabel('f/Hz')
            plt.ylabel('A')
            plt.xlim([fL[1],fL[-1]/2])
            #plt.gca().semilogx()
            plt.savefig('%s%s%d.jpg'%(outputDir,fileStr,i),dpi=200)
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
    def set(self,modelOld):
        self.set_weights(modelOld.get_weights())
    def setTrain(self,name,trainable=True):
        lr0= K.get_value(self.optimizer.lr)
        for layer in self.layers:
            if layer.name.split('_')[0] in name:
                layer.trainable = trainable
                print('set',layer.name,trainable)
            else:
                layer.trainable = not trainable
                print('set',layer.name,not trainable)

        self.compile(loss=self.config.lossFunc, optimizer='Nadam')
        K.set_value(self.optimizer.lr,  lr0)

tTrain = (10**np.arange(0,1.000001,1/29))*16

def trainAndTest(model,corrLTrain,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,3,2,1.5],count0=3,perN=200):
    '''
    依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    w0 = 3#2#4#8#8*3#8#5#10##model.config.lossFunc.w
    testCount = len(corrLTest)
    showCount = int(len(corrLTest)*3/4)
    showD     = int(showCount/40)
    resStr = 'testCount %d showCount %d \n'%(testCount,showCount)
    for sigma in sigmaL:
        model.config.lossFunc.w = w0*(4/sigma)**0.5
        corrLTrain.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLTest.iL=np.array([])
        model.compile(loss=model.config.lossFunc, optimizer='Nadam')
        xTest, yTest, tTest =corrLTest(np.arange(showCount,testCount))
        resStr+=model.trainByXYT(corrLTrain,xTest=xTest,yTest=yTest,\
            count0=count0, perN=perN)
        
        
    xTest, yTest, tTest =corrLTest(np.arange(showCount))
    yout=model.predict(xTest)
    resStr += '\n test part\n'
    resStr+= printRes_old(yTest, yout)
    resStr+= printRes(yTest, yout)+'\n'
    with open(outputDir+'resStr','w') as f:
        f.write(resStr)
    for threshold in [0.5,0.7,0.8]:
            corrLTest.plotPickErro(yout,tTrain,fileName=outputDir+'erro.jpg',\
                threshold=threshold)
    iL=np.arange(0,showCount,showD)
    model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir)



def trainAndTestCross(model0,model1,corrLTrain0,corrLTrain1,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,2],modeL=['conv','conv']):
    '''
    依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    #different data train different part
    w0 = 5#5#10##model.config.lossFunc.w
    for i in range(len(sigmaL)):
        sigma = sigmaL[i]
        mode = modeL[i]
        model0.config.lossFunc.w = w0*(4/sigma)**0.5
        model1.config.lossFunc.w = w0*(4/sigma)**0.5
        corrLTrain0.timeDisKwarg['sigma']=sigma
        corrLTrain1.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLTest.iL=np.array([])
        if mode =='conv':
            model0.setTrain(['conv','CONV'],True)
            model1.setTrain([],False)
            per1=0.5
        if mode =='anti_conv':
            model0.setTrain([],False)
            model1.setTrain(['conv','CONV'],True)
            per1=0.5
        if mode =='dconv':
            model0.setTrain([],False)
            model1.setTrain(['dconv'],True)
            per1=2
        if mode =='None':
            model0.setTrain([],False)
            model1.setTrain([],False)
            per1=0.5
        if mode =='conv_dconv':
            model0.setTrain(['conv','CONV'],True)
            model1.setTrain(['dconv','DCONV'],True)
            per1 = 0.5
        if mode =='0':
            model0.setTrain([],False)
            model1.setTrain([],True)
            per1 = 0.5
        xTest, yTest, tTest =corrLTest(np.arange(2000,4000))
        model0.trainByXYTCross(model1,corrLTrain0,corrLTrain1,xTest=xTest,yTest=yTest,per1=per1)
    xTest, yTest, tTest =corrLTest(np.arange(2000))
    corrLTest.plotPickErro(model0.predict(xTest),tTrain,\
    fileName=outputDir+'erro.jpg')
    iL=np.arange(0,1000,50)
    model0.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir)
    xTest, yTest, tTest =corrLTrain0(np.arange(10000))
    corrLTrain0.plotPickErro(model0.predict(xTest),tTrain,\
    fileName=outputDir+'erro0.jpg')
    iL=np.arange(0,1000,50)
    model0.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir+'_0_')
    xTest, yTest, tTest =corrLTrain1(np.arange(10000))
    corrLTrain1.plotPickErro(model0.predict(xTest),tTrain,\
    fileName=outputDir+'erro1.jpg')
    iL=np.arange(0,1000,50)
    model0.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir+'_1_')


    
'''

for i in range(10):
    plt.plot(inputData[i,:,0,0]/5,'k',linewidth=0.3)
    plt.plot(probP[i,:,0,0].transpose(),'b',linewidth=0.3)
    plt.plot(probS[i,:,0,0].transpose(),'r',linewidth=0.3)
    plt.show()
'''
