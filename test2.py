import dispersion as d
import fk
from imp import reload
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import scipy
from scipy import io as sio
import os
import mathFunc
import random
import fcn
import h5py
import seism
#是否需要考虑展平变化的影响
orignExe='/home/jiangyr/program/fk/'
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
srcSacDirTest='/home/jiangyr/Surface-Wave-Dispersion/srcSacTest/'
T=np.array([0.5,1,2,5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250,275,300])

para={'freq'      :[1/180,1/10]}
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=1000,maxDist=1e8,minDDist=200,\
        maxDDist=3000,para=para,isFromO=True,removeP=True)
configTest=d.config(originName='models/ak135',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=1000,maxDist=1e8,minDDist=200,maxDDist=3000,para=para,\
        isFromO=True,removeP=True)

stations = seism.StationList('stations/NEsta_all.locSensorDas')
stations.getSensorDas()
stations.getInventory()

fvNED = config.loadNEFV(stations)
config.plotFVL(fvNED,pog='p')
fvALLD = {}
fvALLD.update(fvNED)
fvALLD.update(fvPD)
fvALLD.update(fvPDTest)
f = fk.FK(orignExe=orignExe)

pN = 5
pL=[]
for i in range(pN):
    pL.append(  multiprocessing.Process(target=config.calFv, args=(range(i,1000,pN),'p')))
    pL.append(  multiprocessing.Process(target=config.calFv, args=(range(i,1000,pN),'g')))
    pL.append(  multiprocessing.Process(target=configTest.calFv, args=(range(i,1000,pN),'p')))
    pL.append(  multiprocessing.Process(target=configTest.calFv, args=(range(i,1000,pN),'g')))


for p in pL:
    p.start()

for p in pL:
    p.join()



tTrain = np.array([5,10,20,30,50,80,100,150,200,250])
tTrain = np.array([5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250])
tTrain = (10**np.arange(0,1.000001,1/29))*16


i = 0

corrLP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0)
corrLTestP.setTimeDis(fvPDTest,tTrain,sigma=4,\
maxCount=4096,byT=False,noiseMul=0.0)

modelP = fcn.model(channelList=[0])
fcn.trainAndTest(modelP,corrLP,corrLTestP,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)

stationsTrain = seism.StationList('stations/NEsta_all.locSensorDas')
stationsTrain.getSensorDas()
stationsTrain.getInventory()

quakesTrain   = seism.QuakeL('phaseLNE')
corrLQuakeP = d.corrL(config.quakeCorr(quakes[:],stationsTrain,\
    False,remove_resp=True,minSNR=40,isLoadFv=True,fvD=fvNED))
corrLTrain     =  d.corrL(corrLQuakeP+corrLP)
corrLTrain.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0)
stationsTrain = seism.StationList('stations/NEsta_all.locSensorDas')
stationsTrain.getSensorDas()
stationsTrain.getInventory()
fvALLD = fvNED+{}
quakesTrain   = seism.QuakeL('phaseLNE')
corrLQuakeP = d.corrL(config.quakeCorr(quakesTrain[:],stationsTrain,\
    False,remove_resp=True,minSNR=40,isLoadFv=True,fvD=fvNED))
corrLTest     =  d.corrL(corrLQuakeP+[])
corrLTest.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0)

fcn.trainAndTest(modelP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)

xQuake, yQuake, tQuake =corrLQuakeP(np.arange(0,400,10))
modelP.show(xQuake, yQuake,time0L=tQuake,delta=1.0,T=tTrain,\
        outputDir='predict/R_P')
corrLQuakePNew = d.corrL(corrLQuakeP[0:40000:10])
corrLQuakePNew.setTimeDis(fvPD,tTrain,sigma=1.5,maxCount=4096,byT=False)
corrLQuakePNew.getAndSave(modelP,'predict/v_probQuakeP',isPlot=True,\
    isSimple=False,D=0.2)


corrLQuakeG = corrLQuakeP.copy()#d.corrL(config.quakeCorr(quakes[:10],stations,False))
corrLQuakeG.getTimeDis(fvGD,tTrain,sigma=4,maxCount=4096,\
    randD=30,byT=False)
iL=np.arange(0,10000,250)
modelG.show(corrLQuakeG.x[iL],corrLQuakeG.y[iL],\
        time0L=corrLQuakeG.t0L[0:10000:250],delta=1.0,T=tTrain,\
        outputDir='predict/R_G')


