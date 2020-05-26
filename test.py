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
'''
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(400,1800,100),srcSacNum=100,delta=0.5,layerN=20,\
        layerMode='prem',getMode = 'norm',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=150,\
        xcorrFuncL = [mathFunc.xcorrSimple,mathFunc.xcorrComplex],isFlat=True,R=6371,flatM=-2,pog='p',calMode='fast',\
        T=T,threshold=0.1,expnt=10,dk=0.1,\
        fok='/k')
configTest=d.config(originName='models/ak135',srcSacDir=srcSacDir,\
        distance=np.arange(400,1800,100),srcSacNum=100,delta=0.5,layerN=28,\
        layerMode='prem',getMode = 'norm',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=150,\
        xcorrFuncL = [mathFunc.xcorrSimple,mathFunc.xcorrComplex],isFlat=True,R=6371,flatM=-2,pog='p',calMode='fast',\
        T=T,threshold=0.1,expnt=10,dk=0.1,\
        fok='/k')
        '''
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=15,isCut=False,\
        minDist=1000,maxDist=1e8,minDDist=200,maxDDist=3000)
configTest=d.config(originName='models/ak135',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=15,isCut=False,\
        minDist=1000,maxDist=1e8,minDDist=200,maxDDist=3000)

#config.genModel(N=1000,perD= 0.30,depthMul=2)
#configTest.genModel(N=1000,perD= 0.30,depthMul=2)
pL = []
f = fk.FK(orignExe=orignExe)
#fk.genSourceSacs(f,config.srcSacNum,config.delta,srcSacDir = srcSacDir,time=80)
#fk.genSourceSacs(f,config.srcSacNum,config.delta,srcSacDir = srcSacDirTest,time=80)

pN = 5
for i in range(pN):
    pL.append(  multiprocessing.Process(target=config.calFv, args=(range(i,1000,pN),'p')))
    pL.append(  multiprocessing.Process(target=config.calFv, args=(range(i,1000,pN),'g')))
    pL.append(  multiprocessing.Process(target=configTest.calFv, args=(range(i,1000,pN),'p')))
    pL.append(  multiprocessing.Process(target=configTest.calFv, args=(range(i,1000,pN),'g')))


for p in pL:
    p.start()

for p in pL:
    p.join()

fkL = fk.fkL(20,exePath='FKRUN/',orignExe=orignExe,resDir='FKRES/')
FKCORR  = d.fkcorr(config)
corrLL  = fkL(1000,FKCORR)
FKCORRTest  = d.fkcorr(configTest)
corrLLTest  = fkL(200,FKCORRTest)

corrMatL = []
corrMatTestL = []


fvGD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_g_0'%i,'file')for i in range(1000)}
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_p_0'%i,'file')for i in range(1000)}
fvGDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_g_0'%i,'file')for i in range(1000)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_p_0'%i,'file')for i in range(1000)}
fvGD['models/prem']= d.fv('models/prem_fv_flat_new_g_0','file')
fvPD['models/prem']= d.fv('models/prem_fv_flat_new_p_0','file')
fvGD['models/ak135']= d.fv('models/ak135_fv_flat_new_g_0','file')
fvPD['models/ak135']= d.fv('models/ak135_fv_flat_new_p_0','file')
disDir = 'disDir/'
mL = [config.getModel('models/prem%d'%i)for i in range(1000)]
mLTest = [configTest.getModel('models/prem%d'%i)for i in range(1000)]
config.plotModelL(mL)
config.plotFVL(fvPD,'p')
config.plotFVL(fvGD,'g')
configTest.plotModelL(mLTest)
configTest.plotFVL(fvPDTest,'p')
configTest.plotFVL(fvGDTest,'g')
if not os.path.exists(disDir):
    os.makedirs(disDir)

tTrain = np.array([5,10,20,30,50,80,100,150,200,250])
tTrain = np.array([5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250])
def trainAndTest(model,corrLTrain,corrLTest,outputDir='predict/',tTrain=tTrain):
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    xTest, yTest, tTest =corrLTest(np.arange(3000,6000))
    #print(xTest.shape,yTest.shape)
    model.trainByXYT(corrLTrain,xTest=xTest,yTest=yTest)
    #model.train(xTrain, yTrain,xTest=xTest,yTest=yTest)
    xTest, yTest, tTest =corrLTest(np.arange(3000))
    corrLTest.plotPickErro(model.predict(xTest),tTrain,\
        fileName=outputDir+'erro.jpg')
    iL=np.arange(0,1000,50)
    model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
        T=tTrain,outputDir=outputDir)
    

i = 0
stations = seism.StationList('stations/staLstNMV2SelectNew')
noises=seism.QuakeL('noiseL')
n = config.getNoise(noises,stations,mul=0.4)
n.mul = 0.1
corrLP = d.corrL(config.modelCorr(1000,noises=n,randDrop=0.3))
#corrLP = d.corrL(config.modelCorr(200,randDrop=0.3))
corrLTestP = d.corrL(configTest.modelCorr(100,noises=n,randDrop=0.2))
corrLG     = corrLP.copy()
corrLTestG = corrLTestP.copy()


corrLP = d.corrL(corrLP)
corrLTestP = d.corrL(corrLTestP)
corrLP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0)#,self1=corrLP1)
corrLTestP.setTimeDis(fvPDTest,tTrain,sigma=4,\
maxCount=4096,byT=False,noiseMul=0.0)#,self1=corrLTestP1)
corrLG.setTimeDis(fvGD,tTrain,sigma=6,maxCount=4096,\
noiseMul=0.0)#,self1=corrLG1)
corrLTestG.setTimeDis(fvGDTest,tTrain,sigma=6,\
maxCount=4096,noiseMul=0.0)#,self1=corrLTestG1)


modelP = fcn.model(channelList=[0,2,3])
modelG = fcn.model(channelList=[0,2,3])
trainAndTest(modelP,corrLP,corrLTestP,outputDir='predict/P_')
trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')
#trainAndTest(modelP,corrLP,corrLP,outputDir='predict/P_')
#trainAndTest(modelG,corrLG,corrLG,outputDir='predict/G_')
#trainAndTest(modelP,corrLP,corrLP,outputDir='predict/P_')
#trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')
quakes   = seism.QuakeL('phaseL')
corrLQuakeP = d.corrL(config.quakeCorr(quakes[:50],stations,\
    False,para={}))
corrLQuakeP= d.corrL(corrLQuakeP[:5000])
corrLQuakeP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,byT=False)
xQuake, yQuake, tQuake =corrLQuakeP(np.arange(0,6400,160))
modelP.show(xQuake, yQuake,time0L=tQuake,delta=1.0,T=tTrain,\
        outputDir='predict/R_P')

corrLQuakeG = corrLQuakeP.copy()#d.corrL(config.quakeCorr(quakes[:10],stations,False))
corrLQuakeG.getTimeDis(fvGD,tTrain,sigma=4,maxCount=2048,\
    randD=30,byT=False)
iL=np.arange(0,10000,250)
modelG.show(corrLQuakeG.x[iL],corrLQuakeG.y[iL],\
        time0L=corrLQuakeG.t0L[0:10000:250],delta=1.0,T=tTrain,\
        outputDir='predict/R_G')


import seism
from obspy import UTCDateTime


#stations.write('staLstAllNew')
#quakes   = seism.QuakeL('phaseLstVNM_20200305V1')
#quakes.write('phaseL')
stations = seism.StationList('stations/NEsta_all.loc')
quakes   = seism.QuakeL('phaseGlobal')
req ={\
'loc0':stations.loc0(),\
'maxDist':10000,\
'minDist':500,\
'time0':UTCDateTime(2009,1,1).timestamp+243*86400,\
'time1':UTCDateTime(2011,1,1).timestamp+220*86400\
}
quakes.select(req)
quakes.write('phaseLNE')
quakes   = seism.QuakeL('phaseLNE')
para={\
'delta0' :1,
'freq'   :[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e10,
}
quakes[:50].cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False)


stations = seism.StationList('stations/staLstNMV2SelectNew')
stations.getSensorDas()
stations.write('stations/staLstNMV2SelectNewSensorDas',\
    'net sta compBase la lo dep sensorName dasName sensorNum')
quakes   = seism.QuakeL('phaseGlobal')
req ={\
'loc0':stations.loc0(),\
'maxDist':10000,\
'minDist':500,\
'time0':UTCDateTime(2014,1,1).timestamp,\
'time1':UTCDateTime(2017,1,1).timestamp\
}
quakes.select(req)
quakes.write('phaseL')
quakes   = seism.QuakeL('phaseL')

para={\
'delta0' :1,
'freq'   :[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e10,
}
quakes[:100].cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False)


quakes   = seism.QuakeL('phaseL')
noises = quakes.copy()
for noise in noises:
    noise['time']-=5000
noises.write('noiseL')
noises=seism.QuakeL('noiseL')
noises[:10].cutSac(stations,bTime=-10,eTime =4096,para=para,byRecord=False)
noises[:10].getSacFiles(stations)
quakes[:10].getSacFiles(stations)

