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
import sys
sys.path.append("..")
from fcn import genModel0
from tensorflow.keras import backend as K
import h5py
import tensorflow as tf
import logging
import sacTool
import random
os.environ["MKL_NUM_THREADS"] = "32"
fileDir='/home/jiangyr/accuratePickerV3/testNew/'
isBadPlus=1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session =tf.Session(config=config)
K.set_session(session) 
def loadModel(file,mode='norm',phase='p'):
    m = genModel0(mode,phase)
    m.load_weight(file)
    return m
class modelPhase:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(processX(X))


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


def processX(X, rmean=True, normlize=False, reshape=True,isNoise=False,num=2000):
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
    i0=250
    i1=1750
    if num ==2000:
        validL=np.where((maxYIndex-i0)*(maxYIndex-i1)<0)[0]
        tmpY=tmpY[validL]
        tmpY0=tmpY0[validL]

        #print(validL)
        di=(tmpY.reshape([-1,2000])[:, i0:i1].argmax(axis=1)-\
                tmpY0.reshape([-1,2000])[:, i0:i1].argmax(axis=1))
        validL=np.where(np.abs(di)<threshold)[0]
        pTmp=tmpY.reshape([-1,2000])[:, i0:i1].max(axis=1)[validL]
        validLNew=np.where(pTmp>minY)[0]
        validL=validL[validLNew]
        if len(di)==0:
            return 0, 0, 0, 0
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
            return 0, 0, 0,0
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
            return 0, 0, 0,0
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
            return 0, 0, 0,0

    return np.size(validL)/np.size(di),di[validL].mean(),di[validL].std(),len(di)




def train(modelFile, resFile, phase='p',validWN=5000,testWN=10000,\
    validNN=5000,testNN=5000,inN=2000,trainWN=10000,trainNN=2000,\
    modelType='norm',\
    waveFile='/media/jiangyr/MSSD/waveforms_11_13_19.hdf5',\
    catalogFile1='data/metadata_11_13_19.csv'\
    ,catalogFile2='phaseDir/hinetFileLstNew'):
    rms0=1e5
    resCount=20
    logger=logging.getLogger(__name__)
    model,dIndex,channelN= genModel0(modelType,phase)
    logger.info('model mode: %s'%(model.config.mode))
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
    #random.shuffle(catalog1)
    #random.shuffle(catalog2)
    catalogValid=[]
    catalogTest=[]
    catalogTrain=[]
    for  catalog in [catalog1,catalog2]:
        catalogValid+=catalog[:validWN]+catalog[-validNN:]
    for  catalog in [catalog1,catalog2]:
        catalogTest+=catalog[validWN:(testWN+validWN)]\
        +catalog[-(testNN+validNN):-validNN]
    for  catalog in [catalog1,catalog2]:
        #catalogTrain+=catalog[validWN+testWN:(trainWN+testWN+validWN)]\
        #+catalog[-(trainNN+testNN+validNN):-(testNN+validNN)]
        catalogTrain+=catalog[validWN+testWN:-(testNN+validNN)]

    logger.info('vaild num: %d   testNum: %d  trainNum: %d inN: %d'\
        %(len(catalogValid),len(catalogTest),len(catalogTrain),inN))

    xValid,yValid,modeValid=sacTool.getXYFromCatalogP(catalogValid,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase,oIndex=-2)
    xValid=processX(xValid,isNoise=False,num=dIndex)

    
    increaseCount =10
    for i in range(5000):
        catalogIn=random.sample(catalogTrain,inN)
        xTrain,yTrain,modeTrain=sacTool.getXYFromCatalogP(catalogIn,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase)
        xTrain=processX(xTrain,isNoise=False,num=dIndex)
        ne =3
        if i >3:
            ne =1
        if i >6 and i%int(increaseCount)==0:
            K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) \
            * 0.9)#0.95
            increaseCount*=1.05
        bs = 100
        tmpI=i%xTrain.shape[0]
        showXY(xTrain[tmpI],yTrain[tmpI],np.arange(min(yTrain.shape[-1],2)))
        plt.title(modeTrain[tmpI])
        plt.savefig('fig/train_%s/%d_train.jpg'%(phase,i),dpi=300)
        plt.close()
        model.fit(xTrain,yTrain,batchSize=bs)
        logger.info('loop %d runSample/allSample: %.7f'%(i,inN*(i+1)/len(catalogTrain)))
        if i%1==0:
            thresholds = [50, 25, 5]
            minYL=[0.1,0.5,0.9]
            tmpY=model.predict(xValid)
            print(tmpY.shape)
            tmpI=i%xValid.shape[0]
            showXY(xValid[tmpI],tmpY[tmpI],np.arange(min(yTrain.shape[-1],2)))
            plt.title(modeValid[tmpI])
            plt.savefig('fig/train_%s/out_%d_train.jpg'%(phase,i),dpi=300)
            plt.close()
            for threshold in thresholds:
                for minY in minYL:
                    for cI in channelIndex:
                        if cI==2:
                            continue
                        p,m,s,num=validStd(tmpY[:,:,:,channelIndex.tolist().index(cI)],\
                            yValid[:,:,:,channelIndex.tolist().index(cI)], threshold=\
                            threshold, minY=minY,num=dIndex)
                        logger.info('STEAD channel: %d % 3d : minY:%.2f p:\
                            %.5f m:%.5f s:%.5f num: %7d'%(cI,threshold,minY,p,m,s,num))

            p,absMean,rms,num=validStd(tmpY[:,:,:,0],yValid[:,:,:,0]\
                ,threshold=20, minY=0.5,num=dIndex)
            rms=model.evaluate(x=xValid, y=yValid)
            logger.info('vaild loss: %.9f'%rms)
            rms-=p*10
            logger.info('vaild rms: %.9f'%rms)
            if rms >= rms0 and p > 0.45 :
                resCount = resCount-1
                if resCount == 0:
                    model.set_weights(model0)
                    logger.info('over fit ,force to stop, set to best model')
                    break
            if rms < rms0 and p > 0.45 :
                resCount = 50
                rms0 = rms
                model0 = (model.get_weights())
                logger.info('find a better model')
    model.set_weights(model0)
    model.save(modelFile)
    minYL=[0.1,0.5,0.9]
    thresholds = [50, 25, 5]
    xTest,yTest,modeTest=sacTool.getXYFromCatalogP(catalogTest,w,dIndex=dIndex,\
        channelIndex=channelIndex,phase=phase,oIndex=-2)
    xTest=processX(xTest,isNoise=False,num=dIndex)
    outY = model.predict(xTest)
    for threshold in thresholds:
        for minY in minYL:
            for cI in channelIndex:
                if cI==2:
                    continue
                p,m,s,num=validStd(outY[:,:,:,channelIndex.tolist().index(cI)],\
                    yTest[:,:,:,channelIndex.tolist().index(cI)],\
                 threshold=threshold, minY=minY,num=dIndex)
                logger.info('test STEAD channelP:%d  % 3d : minY:%.2f \
                    p:%.5f m:%.5f s:%.5f num:%7d'%(cI,threshold,minY,p,m,s,num))
                
    sio.savemat(resFile, {'out'+phase+'y': outY, 'out'+phase+'x': xTest, \
            phase+'y'+'0': yTest})

def showXY(x,y,channelL):
    A = x.std()*3
    for i in range(3):
        plt.plot(x[:,:,i]/A+1+i*3,'k',linewidth=0.3)
    for i in channelL:
        plt.plot(y[:,:,i]-i-1,linewidth=0.3)


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
'''
inx
inx done
(15551, 2000, 1, 1)
2020-11-09 03:02:46,309 - __main__ - INFO -            STEAD channel: 0  50 : minY:0.10 p:                            0.9869 m:0.6185 s:4.7623
2020-11-09 03:02:46,489 - __main__ - INFO -            STEAD channel: 0  50 : minY:0.50 p:                            0.9815 m:0.6016 s:4.6684
2020-11-09 03:02:46,681 - __main__ - INFO -            STEAD channel: 0  50 : minY:0.90 p:                            0.8906 m:0.4825 s:3.5416
2020-11-09 03:02:46,864 - __main__ - INFO -            STEAD channel: 0  25 : minY:0.10 p:                            0.9803 m:0.6307 s:3.6111
2020-11-09 03:02:47,047 - __main__ - INFO -            STEAD channel: 0  25 : minY:0.50 p:                            0.9753 m:0.6329 s:3.5882
2020-11-09 03:02:47,245 - __main__ - INFO -            STEAD channel: 0  25 : minY:0.90 p:                            0.8877 m:0.4929 s:2.8190
2020-11-09 03:02:47,427 - __main__ - INFO -            STEAD channel: 0   5 : minY:0.10 p:                            0.8713 m:0.1311 s:1.5278
2020-11-09 03:02:47,605 - __main__ - INFO -            STEAD channel: 0   5 : minY:0.50 p:                            0.8680 m:0.1316 s:1.5282
2020-11-09 03:02:47,790 - __main__ - INFO -            STEAD channel: 0   5 : minY:0.90 p:                            0.8165 m:0.1370 s:1.4864
15551/15551 [==============================] - 15s 964us/step
2020-11-09 03:03:04,944 - __main__ - INFO -            over fit ,force to stop, set to best model
inx
inx done
2020-11-09 03:03:58,125 - __main__ - INFO -            test STEAD channelP:0   50 : minY:0.10                     p:0.9906 m:-0.5220 s:4.7276
2020-11-09 03:03:58,411 - __main__ - INFO -            test STEAD channelP:0   50 : minY:0.50                     p:0.9873 m:-0.5170 s:4.6597
2020-11-09 03:03:58,709 - __main__ - INFO -            test STEAD channelP:0   50 : minY:0.90                     p:0.9186 m:-0.5172 s:3.5163
2020-11-09 03:03:58,998 - __main__ - INFO -            test STEAD channelP:0   25 : minY:0.10                     p:0.9825 m:-0.5602 s:3.5593
2020-11-09 03:03:59,282 - __main__ - INFO -            test STEAD channelP:0   25 : minY:0.50                     p:0.9796 m:-0.5614 s:3.5382
2020-11-09 03:03:59,581 - __main__ - INFO -            test STEAD channelP:0   25 : minY:0.90                     p:0.9156 m:-0.5439 s:2.9674
2020-11-09 03:03:59,866 - __main__ - INFO -            test STEAD channelP:0    5 : minY:0.10                     p:0.8719 m:-0.5605 s:1.6176
2020-11-09 03:04:00,154 - __main__ - INFO -            test STEAD channelP:0    5 : minY:0.50                     p:0.8703 m:-0.5615 s:1.6162
2020-11-09 03:04:00,485 - __main__ - INFO -            test STEAD channelP:0    5 : minY:0.90                     p:0.8329 m:-0.5678 s:1.5980
'''
