import argparse
import matplotlib.pyplot as plt
import obspy
import sys
import math
import scipy.io as sio
import scipy
import numpy as np
from numpy import cos, sin
import os
#from genMV3 import genModel0
from models import genModel0
#from keras import backend as K
#from keras.models import Model
import h5py
#import tensorflow as tf
import logging
#from sacTool import getDataByFileName
import sacTool
#from glob import glob
#import obspy
import random
os.environ["MKL_NUM_THREADS"] = "10"
fileDir='/home/jiangyr/accuratePickerV3/testNew/'
isBadPlus=1
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session =tf.Session(config=config)
K.tensorflow_backend.set_session(session) 
'''

class modelPhase:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(processX(X))
    def predictLongData(self, x, N=2000, indexL=range(750, 1250)): 
        return predictLongData(self, x, N=N, indexL= indexL)


def predict(model, X):
    return model.predict(processX(X))


def predictLongData(model, x, N=2000, indexL=range(750, 1250)):
    if len(x) == 0:
        return np.zeros(0)
    N = x.shape[0]
    Y = np.zeros(N)
    perN = len(indexL)
    loopN = int(math.ceil(N/perN))
    perLoop = int(1000)
    inMat = np.zeros((perLoop, 2000, 1, 3))
    for loop0 in range(0, int(loopN), int(perLoop)):
        loop1 = min(loop0+perLoop, loopN)
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                inMat[loop-loop0, :, :, :] = processX(x[sIndex: sIndex+2000, :])\
                .reshape([2000, 1, 3])
        outMat = model.predict(inMat).reshape([-1, 2000])
        for loop in range(loop0, loop1):
            i = loop*perN
            sIndex = min(max(0, i), N-2000)
            if sIndex > 0:
                Y[indexL[0]+sIndex: indexL[-1]+1+sIndex] = \
                outMat[loop-loop0, indexL].reshape([-1])
    return Y


def processX(X, rmean=True, normlize=True, reshape=True,isNoise=False,num=2000):
    if reshape:
        X = X.reshape(-1, num, 1, 3)
    if rmean:
        X-= X.mean(axis=1,keepdims=True)
    if normlize:
        X /=(X.std(axis=(1, 2, 3),keepdims=True))
    if isNoise:
        X+=(np.random.rand(X.shape[0],num,1,3)-0.5)*np.random.rand(X.shape[0],1,1,3)*X.max(axis=(1,2,3),keepdims=True)*0.15*(np.random.rand(X.shape[0],1,1,1)<0.1)
    return X


def processY(Y):
    return Y.reshape(-1, 2000, 1, 1)


def validStd(tmpY,tmpY0,threshold=100, minY=0.2,num=2000):
    tmpY=tmpY.reshape((-1,num))
    tmpY0=tmpY0.reshape((-1,num))
    maxY0=tmpY0.max(axis=1);
    validL=np.where(maxY0>0.9)[0]
    tmpY=tmpY[validL]
    tmpY0=tmpY0[validL]
    maxYIndex=tmpY0.argmax(axis=1)
    if num ==2000:
        validL=np.where((maxYIndex-250)*(maxYIndex-1750)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,2000])[:, 250:1750].argmax(axis=1)-\
                tmpY0.reshape([-1,2000])[:, 250:1750].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,2000])[:, 250:1750].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0
    if num==1600:
        validL=np.where((maxYIndex-200)*(maxYIndex-1400)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,num])[:, 200:1400].argmax(axis=1)-\
                tmpY0.reshape([-1,num])[:, 200:1400].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,1600])[:, 200:1400].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0
    if num==1200:
        validL=np.where((maxYIndex-200)*(maxYIndex-1000)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,num])[:, 200:1000].argmax(axis=1)-\
                tmpY0.reshape([-1,num])[:, 200:1000].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,1200])[:, 200:1000].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0
    if num==1500:
        validL=np.where((maxYIndex-250)*(maxYIndex-1250)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,num])[:, 250:1250].argmax(axis=1)-\
                tmpY0.reshape([-1,num])[:, 250:1250].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,1500])[:, 250:1250].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0

    return np.size(validL)/np.size(di),di[validL].mean(),di[validL].std()




def train(modelFile, resFile, phase='p',validWN=500,testWN=500,\
    validNN=100,testNN=100,inN=2000,trainWN=1000,trainNN=200,\
    modelType='norm',\
    waveFile='data/waveforms_11_13_19.hdf5',\
    catalogFile1='data/metadata_11_13_19.csv'\
    ,catalogFile2='phaseDir/hinetFileLst'):
    rms0=1e5
    resCount=20
    logger=logging.getLogger(__name__)
    model,dIndex,channelN= genModel0(modelType,phase)
    print(model,dIndex,channelN)
    model0 = model.get_weights()
    print(model.summary())
    if phase=='p':
        channelIndex=np.arange(0,1)
    if phase=='s':
        channelIndex=np.arange(1,2)
    if phase=='ps':
        channelIndex=np.arange(3)
    w = h5py.File(waveFile,'r')

    catalog1,d1=sacTool.getCatalogFromFile(catalogFile1,mod='STEAD')
    catalog2,d2=sacTool.getCatalogFromFile(catalogFile2,mod='hinet')
    
    catalogValid=[]
    catalogTest=[]
    catalogTrain=[]
    for  catalog in [catalog1,catalog2]:
        catalogValid+=catalog[:validWN]+catalog[-validNN:]
        catalogTest+=catalog[validWN:(testWN+validWN)]\
        +catalog[-(testNN+validNN):-validNN]
        catalogTrain+=catalog[validWN+testWN:(trainWN+testWN+validWN)]\
        +catalog[-(trainNN+testNN+validNN):-(testNN+validNN)]

    logger.info('vaild num: %d   testNum: %d  trainNum: %d inN: %d'\
        %(len(catalogValid),len(catalogTest),len(catalogTrain),inN))

    xValid,yValid=sacTool.getXYFromCatalog(catalogValid,w,dIndex=dIndex,\
        channelIndex=channelIndex)
    xValid=processX(xValid,isNoise=False,num=dIndex)

    xTest,yTest=sacTool.getXYFromCatalog(catalogTest,w,dIndex=dIndex,\
        channelIndex=channelIndex)
    xTest=processX(xTest,isNoise=False,num=dIndex)
    
    for i in range(500):
        catalogIn=random.sample(catalogTrain,inN)
        xTrain,yTrain=sacTool.getXYFromCatalog(catalogIn,w,dIndex=dIndex,\
        channelIndex=channelIndex)
        xTrain=processX(xTrain,isNoise=False,num=dIndex)
        ne =3
        if i >3:
            ne =1
        if i >6 and i%3==0:
            #K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) \
            #* 0.9)#0.95
            pass
        bs = 50
        if i > 5/3:
            bs = 75
        if i > 10/3:
            bs = 100
        if i > 20/3:
            bs = 110
        if i > 30/3:
            bs = 120
        if i >40/3:
            bs = 125
        if i > 50/3:
            bs = 130
        if i > 100/3:
            bs = 150
        if i > 150/3:
            bs = 200
        if i > 300/3:
            bs=300
        tmpI=i%xTrain.shape[0]
        showXY(xTrain[tmpI],yTrain[tmpI],np.arange(min(yTrain.shape[-1],2)))
        plt.savefig('fig/train/%d_train.jpg'%i,dpi=300)
        plt.close()
        model.fit(xTrain,yTrain, nb_epoch=ne,  \
         batch_size=bs, verbose=2)
        logger.info('loop %d'%(i))
        if i%1==0:
            thresholds = [50, 25, 5]
            minYL=[0.1,0.5,0.9]
            tmpY=model.predict(xValid, verbose=0)
            print(tmpY.shape)
            tmpI=i%xValid.shape[0]
            showXY(xValid[tmpI],tmpY[tmpI],np.arange(min(yTrain.shape[-1],2)))
            plt.savefig('fig/train/out_%d_train.jpg'%i,dpi=300)
            plt.close()
            for threshold in thresholds:
                for minY in minYL:
                    for cI in channelIndex:
                        if cI==2:
                            continue
                        p,m,s=validStd(tmpY[:,:,:,cI],yValid[:,:,:,cI], threshold=\
                            threshold, minY=minY,num=dIndex)
                        logger.info('STEAD channel: %d % 3d : minY:%.2f p:\
                            %.4f m:%.4f s:%.4f'%(cI,threshold,minY,p,m,s))

            p,absMean,rms=validStd(tmpY[:,:,:,0],yValid[:,:,:,0]\
                ,threshold=20, minY=0.5,num=dIndex)
            rms=model.evaluate(x=xValid, y=yValid)
            rms-=p*100
            if rms >= rms0 and p > 0.45 :
                resCount = resCount-1
                if resCount == 0:
                    model.set_weights(model0)
                    logger.info('over fit ,force to stop, set to best model')
                    break
            if rms < rms0 and p > 0.45 :
                resCount = 20
                rms0 = rms
                model0 = (model.get_weights())
                logger.info('find a better model')
    model.set_weights(model0)
    model.save(modelFile)
    minYL=[0.1,0.5,0.9]
    thresholds = [50, 25, 5]
    outY = model.predict(xTest, verbose=0)
    for threshold in thresholds:
        for minY in minYL:
            for cI in channelIndex:
                if cI==2:
                    continue
                p,m,s=validStd(outY,yTest[:,:,:,cI],\
                 threshold=threshold, minY=minY,num=dIndex)
                logger.info('test STEAD channelP:%d  % 3d : minY:%.2f \
                    p:%.4f m:%.4f s:%.4f'%(cI,threshold,minY,p,m,s))
                
    sio.savemat(resFile, {'out'+phase+'y': outY, 'out'+phase+'x': xTest, \
            phase+'y'+'0': yTest})

def showXY(x,y,channelL):
    plt.plot(x[:,:,2]+1,linewidth=0.3)
    for i in channelL:
        plt.plot(y[:,:,i]-i,linewidth=0.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--phase', '-p', type=str, help='train for p or s')
    parser.add_argument('--Num', '-N', type=int, help='number of JP phase')
    parser.add_argument('--modelType', '-m', type=str, help='isSoft')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, \
        format='%(asctime)s - %(name)s - %(levelname)s -\
            %(message)s')
    phase=args.phase
    WN=args.Num
    NN=int(WN/5)
    modelType=args.modelType
    modelFile='model/%s_%s_%d_%d'%(modelType,phase,WN,NN)
    resFile='resDir/res_%s_%s_%d_%d.mat'%(modelType,phase,WN,NN)
    logger=logging.getLogger(__name__)
    logger.info('doing train')
    logger.info('model type:%s'%modelType)
    logger.info('phase:%s'%phase)
    logger.info('Phase num:%d Noise num:%d'%(WN,NN))
    train(modelFile,resFile,trainWN=WN,trainNN=NN,phase=phase,\
        modelType=modelType)
