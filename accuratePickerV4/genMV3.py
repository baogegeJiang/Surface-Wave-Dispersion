from tensorflow import keras
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, MaxPooling2D,\
  AveragePooling2D,Conv2D,Conv2DTranspose,concatenate,\
  Dropout,BatchNormalization, Dense
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.keras import initializers, regularizers, constraints, activations
#LayerNormalization = keras.layers.BatchNormalization
import numpy as np
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt   
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
import random
import obspy
import time
import os
from tensorflow.keras.utils import plot_model

w1=np.ones(1500)*0.5
w0=np.ones(250)*(-0.75)
w2=np.ones(250)*(-0.25)
w=np.append(w0,w1)
w=np.append(w,w2)
wY=K.variable(w.reshape((1,2000,1,1)))

w11=np.ones(1500)*0
w01=np.ones(250)*(-0.75)*0
w21=np.ones(250)*(-0.25)*0
w1=np.append(w01,w11)
w1=np.append(w1,w21)
W1=w1.reshape((1,2000,1,1))
wY1=K.variable(W1)
wY1Short=K.variable(W1[:,200:1800])
wY1Shorter=K.variable(W1[:,400:1600])
wY1500=K.variable(W1[:,250:1750])
W2=np.zeros((1,2000,1,3))
W2[0,:,:,0]=W1[0,:,:,0]*0+(1-0.13)
W2[0,:,:,1]=W1[0,:,:,0]*0+(1-0.13)
W2[0,:,:,2]=W1[0,:,:,0]*0+0.13
wY2=K.variable(W2)
def inAndOutFuncNewV6(config, onlyLevel=-10000):
    BNA = -1
    inputs  = Input(config.inputSize,name='inputs')
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
        
        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'0',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'0')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'0')(last)

        convL[i] =last

        last = Conv2D(config.featureL[i],kernel_size=config.kernelL[i],\
            strides=(1,1),padding='same',name=name+layerStr+'1',\
            kernel_initializer=config.initializerL[i],\
            bias_initializer=config.bias_initializerL[i])(last)

        if i in config.dropOutL:
            ii   = config.dropOutL.index(i)
            last =  Dropout(config.dropOutRateL[ii],name='Dropout'+layerStr+'0')(last)
        else:
            last = BatchNormalization(axis=BNA,trainable=True,name='BN'+layerStr+'1')(last)

        last = Activation(config.activationL[i],name='AC'+layerStr+'1')(last)

        last = config.poolL[i](pool_size=config.strideL[i],\
            strides=config.strideL[i],padding='same',name='PL'+layerStr+'0')(last)

    convL[depth] =last
    outputsL =[]
    for i in range(depth-1,-1,-1):
        if i <3:
            name = 'dconv'
        else:
            name = 'DCONV'
        
        for j in range(i+1):

            layerStr='_%d_%d'%(i,j)

            dConvL[j]= Conv2DTranspose(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=config.strideL[j],padding='same',name=name+layerStr+'0',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(convL[j+1])

            if j in config.dropOutL:
                jj   = config.dropOutL.index(j)
                dConvL[j] =  Dropout(config.dropOutRateL[jj],name='Dropout_'+layerStr+'0')(dConvL[j])
            else:
                dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'0')(dConvL[j])

            dConvL[j]  = Activation(config.activationL[j],name='Ac_'+layerStr+'0')(dConvL[j])
            dConvL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'0')
            dConvL[j]  = Conv2D(config.featureL[j],kernel_size=config.kernelL[j],\
                strides=(1,1),padding='same',name=name+layerStr+'1',\
                kernel_initializer=config.initializerL[j],\
                bias_initializer=config.bias_initializerL[j])(dConvL[j])
            dConvL[j] = BatchNormalization(axis=BNA,trainable=True,name='BN_'+layerStr+'1')(dConvL[j])
            dConvL[j] = Activation(config.activationL[j],name='Ac_'+layerStr+'1')(dConvL[j])
            convL[j]  = concatenate([dConvL[j],convL[j]],axis=BNA,name='conc_'+layerStr+'1')
            if i <config.deepLevel and j==0:
                #outputsL.append(Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
                #padding='same',activation='sigmoid',name='dconv_out_%d'%i)(convL[0]))
                outputsL.append(Dense(config.outputSize[-1], activation='sigmoid'\
                    ,name='dense_out_%d'%i)(convL[0]))
        
    #outputs = Conv2D(config.outputSize[-1],kernel_size=(8,1),strides=(1,1),\
    #    padding='same',activation='sigmoid',name='dconv_out')(convL[0])
    if len(outputsL)>1:
        outputs = concatenate(outputsL,axis=2,name='lastConc')
    else:
        outputs = outputsL[-1]
    if onlyLevel>-100:
        outputs = outputsL[onlyLevel]
    return inputs,outputs

def lossFunc(y,yout):
    return -K.mean((y*K.log(yout+1e-9)+(1-y)*(K.log(1-yout+1e-9)))*(y*0+1)*(1+K.sign(y)*wY),axis=[0,1,2,3])

def lossFuncNew(y,yout):

    yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1)*yW,axis=[0,1,2,3])

def lossFuncNewShort(y,yout):

    yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1Short)*yW,axis=[0,1,2,3])

def lossFuncNewShorter(y,yout):

    yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1Shorter)*yW,axis=[0,1,2,3])

def lossFuncNew1500(y,yout):

    yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1500)*yW,axis=[0,1,2,3])

def lossFuncNewS(y,yout):
    y=y
    yW=(K.sign(-y-0.1)+1)*10*(K.sign(yout-0.35)+1)+1
    y=(K.sign(y+0.1)+1)*y/2
    y0=0.13
    return -K.mean((y*K.log(yout+1e-9)/y0+(1-y)*(K.log(1-yout+1e-9))/(1-y0))*(y*0+1)*(1+K.sign(y)*wY1)*yW,axis=[0,1,2,3])

def lossFuncSoft(y,yout):
    return -K.mean(wY2*y*K.log(yout+1e-9),axis=[0,1,2,3])#1e-6
    #return -K.sum(K.mean(yNew*K.log(youtNew+1e-9),axis=[0,1,2])/K.mean(yNew,axis=[0,1,2]),axis=-1)

def genModel0(modelType='norm',phase='p'):

    if modelType=='norm':
        return genModel(phase),2000,1
    if modelType=='soft':
        return genModelSoft(phase),2000,3
    if modelType=='short':
        return genModelShort(phase),1600,1
    if modelType=='shorter':
        return genModelShorter(phase),1200,1
    if modelType=='1500':
        return genModel1500(phase),1500,1
def genModel(phase='p'):
    inputs, outputA=inAndOutFuncNewV6(config, onlyLevel=-10000)
    model=Model(inputs=inputs,outputs=outputA)
    if phase=='p':
        model.compile(loss=lossFuncNew, optimizer='Nadam')
    else:
        model.compile(loss=lossFuncNewS, optimizer='Nadam')
    return model

