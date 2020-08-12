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

para={'freq'      :[1/150,1/5]}
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,\
        maxDDist=3000,para=para,isFromO=True,removeP=True)
configTest=d.config(originName='models/ak135',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,maxDDist=3000,para=para,\
        isFromO=True,removeP=True)

configSmooth=d.config(originName='models/output_smooth',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=243,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,doFlat=False,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,maxDDist=3000,para=para,\
        isFromO=True,removeP=True)

stations = seism.StationList('stations/NEsta_all.locSensorDas')
stations.getSensorDas()
stations.getInventory()

fvNEDAvarage = config.loadNEFV(stations)
fvNED,quakes0 = config.loadQuakeNEFV(stations)
quakes0=seism.QuakeL(quakes0)
quakes0.write('phaseLPick')
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_p_0'%i,'file')\
for i in range(1000)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_p_0'%i,'file')\
for i in range(1000)}
fvPD['models/prem']= d.fv('models/prem_fv_flat_new_p_0','file')
fvPDTest['models/ak135']= d.fv('models/ak135_fv_flat_new_p_0','file')
#config.plotFVL(fvNED,pog='p')
fvALLD = {}
fvALLD.update(fvNED)
fvALLD.update(fvPD)
fvALLD.update(fvPDTest)
fvALLD.update(fvNEDAvarage)




tTrain = np.array([5,10,20,30,50,80,100,150,200,250])
tTrain = np.array([5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250])
tTrain = (10**np.arange(0,1.000001,1/29))*16
tTrain = (10**np.arange(0,1.000001,1/29))*10


i = 0

stationsN = seism.StationList('stations/staLstNMV2SelectNewSensorDasCheck') + seism.StationList('stations/NEsta_all.locSensorDas')
stationsN.getInventory()
noises=seism.QuakeL('noiseL') + seism.QuakeL('noiseLNE')
n = config.getNoise(noises,stationsN,mul=3,para=para,\
    byRecord=False,remove_resp=True)

n.mul = 0.5
corrLP = d.corrL(config.modelCorr(1000,noises=n,randDrop=0.3,minSNR=0.1))
corrLTestP = d.corrL(configTest.modelCorr(100,noises=n,randDrop=0.2,minSNR=0.1))
corrLP=d.corrL(corrLP)
corrLP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.02,byAverage=True)

stationsTrain = seism.StationList('stations/NEsta_all.locSensorDas')
stationsTrain.getSensorDas()
stationsTrain.getInventory()

quakesTrain   = seism.QuakeL('phaseLPick')
corrLQuakeP = d.corrL(config.quakeCorr(quakesTrain[:],stationsTrain,\
    False,remove_resp=True,minSNR=2,isLoadFv=True,fvD=fvNED,\
    isByQuake=True))
corrLQuakeP0 =corrLQuakeP
corrLQuakeP = d.corrL(corrLQuakeP0,fvD= fvALLD)
corrLTrain0     =  d.corrL(corrLQuakeP[4000:]+corrLP[:10000])
corrLTrain0.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)

corrLTrain1     =  d.corrL(corrLP[:])
corrLTrain1.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)

corrLTest     =  d.corrL(corrLQuakeP[:4000])
corrLTest.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)
corrLTest([1])

modelPReal = fcn.model(channelList=[0,2,3])
modelPSyn = fcn.model(channelList=[0,2,3])
#fcn.trainAndTest(modelP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,3],tTrain=tTrain)
#fcn.trainAndTest2(modelP,corrLP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)
fcn.trainAndTestCross(modelPReal,modelPSyn,corrLTrain0,corrLTrain1,corrLTest,\
    outputDir='predict/P_',sigmaL=[4,3],tTrain=tTrain,modeL=['None','None'])

modelPReal = fcn.model(channelList=[0,2,3])
modelPSyn = fcn.model(channelList=[0,2,3])
#fcn.trainAndTest(modelP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,3],tTrain=tTrain)
#fcn.trainAndTest2(modelP,corrLP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)
fcn.trainAndTestCross(modelPReal,modelPSyn,corrLTrain1,corrLTrain1,corrLTest,\
    outputDir='predict/Syn_P_',sigmaL=[4,3],tTrain=tTrain,modeL=['None','None'])

modelPReal = fcn.model(channelList=[0,2,3])
modelPSyn = fcn.model(channelList=[0,2,3])
#fcn.trainAndTest(modelP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,3],tTrain=tTrain)
#fcn.trainAndTest2(modelP,corrLP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)
fcn.trainAndTestCross(modelPReal,modelPSyn,corrLTrain0,corrLTrain0,corrLTest,\
    outputDir='predict/Real_P_',sigmaL=[4,3],tTrain=tTrain,modeL=['None','None'])
xQuake, yQuake, tQuake =corrLQuakePTest(np.arange(0,400,10))
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





import nb
mp = config.getModel()
NB = nb.NB()
NB.test(mp)
NB.test(nb.Model4())
NB.test(nb.Model3())



config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(300,10000,150),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,\
        maxDDist=3000,para=para,isFromO=True,removeP=True,QMul=1000)

FKCORR = d.fkcorr(config)
FK = fk.fkL(1,exePath='FKRUN/',orignExe=orignExe,resDir='FKRES/')[0]
FKCORR(0,[-1],FK,mul=0,depth0=150,srcSacIndex=-1)

config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,\
        maxDDist=3000,para=para,isFromO=True,removeP=True) 

mL = [config.getModel('models/prem%d'%i) for i in range(200)]
NB = nb.NB(saveDt = 1)

for m in mL[:100]:
    modelName = 'nb_'+os.path.basename(m.modelFile)
    paraNB= {'model':modelName,\
        'strike':360*np.random.rand(),\
        'dip':90*np.random.rand(),\
        'rake':360*np.random.rand()-180,\
        'alpha':-5-int(10*np.random.rand()),\
        'zs':int((20+300*np.random.rand()**2)/NB.para0['h'])}
    NB.isH = False
    N = int(100/NB.para0['dt'])
    duraCount = int((20+60*np.random.rand())/NB.para0['dt'])
    NB.H = np.zeros(N)
    mathFunc.randomSource(int(1000*np.random.rand())%5,\
        duraCount,NB.H)
    NB.test(m,paraNB,isFilter=True,isNoise=False)

for m in mL[0:10]:
    modelName = 'nb_'+os.path.basename(m.modelFile)
    paraNB= {'model':modelName,\
        'strike':360*np.random.rand(),\
        'dip':90*np.random.rand(),\
        'rake':360*np.random.rand()-180,\
        'alpha':-int((10+20*np.random.rand())/NB.para0['dt']),\
        'zs':int((20+300*np.random.rand()**2)/NB.para0['h'])}
    NB.isH = False
    N = int(100/NB.para0['dt'])
    duraCount = int((20+60*np.random.rand())/NB.para0['dt'])
    NB.H = np.zeros(N)
    mathFunc.randomSource(int(1000*np.random.rand())%5,\
        duraCount,NB.H)
    NB.test(m,paraNB,isFilter=True,isNoise=False)


configNB=d.config(originName='models/nb_prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=10000,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,doFlat=False,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,\
        maxDDist=3000,para=para,isFromO=True,removeP=True,modelMode='fileP')
#removeP = True

configNB.calFv([0],'p')
pL = []
pN = 10
for i in range(pN):
    pL.append(  multiprocessing.Process(target=configNB.calFv,\
     args=(range(i,10,pN),'p')))
    pL.append(  multiprocessing.Process(target=configNB.calFv,\
     args=(range(i,200,pN),'g')))

for p in pL:
    p.start()

for p in pL:
    p.join()

iL=[]
for i in range(200):
    if len(glob('models/nb_prem%d_fv_flat_new_p_0_/*'%i))==0:
        iL.append(i)
configNB.calFv(iL, 'p')
fvPDNB = {'models/nb_prem%d'%i: d.fv('models/nb_prem%d_fv_flat_new_p_0'%i,'fileP')\
for i in range(0,10)}
fvALLD.update(fvPDNB)
n.mul=0.2
corrLPNBH =  d.corrL(config.modelCorr(1000,noises=n,randDrop=0.3,minSNR=0.1))#d.corrL(configNB.modelCorr(10,randDrop=0.3,minSNR=0.1))
corrLPNBNH = d.corrL(configNB.modelCorr(np.arange(10),noises=n,randDrop=0.3,minSNR=0.1))

corrLTrain0     =  d.corrL(corrLQuakeP[4000:]+corrLP[:10000]+corrLPNB[:10000])
corrLTrain0.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)

corrLTrain1     =  d.corrL(corrLP[:]+corrLPNB[:])
corrLTrain1.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)

corrLTest     =  d.corrL(corrLQuakeP[:4000])
corrLTest.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)
corrLTest([1])

#corrLPNB .setTimeDis(fvPDNB,tTrain,sigma=4,maxCount=4096,\
#byT=False,noiseMul=0.0,byA=True,rThreshold=0.1,byAverage=True)
modelPReal = fcn.model(channelList=[0,2,3])
modelPSyn = fcn.model(channelList=[0,2,3])
#fcn.trainAndTest(modelP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,3],tTrain=tTrain)
#fcn.trainAndTest2(modelP,corrLP,corrLTrain,corrLTest,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)
fcn.trainAndTestCross(modelPReal,modelPSyn,corrLTrain0,corrLTrain1,corrLTest,\
    outputDir='predict/NB_Real_P_',sigmaL=[4,3],tTrain=tTrain,modeL=['None','None'])


corrLPNBH = corrLP
corrLPNBH.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0,byAverage=True)
corrLPNBNH.setTimeDis(fvALLD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0,byAverage=True)

fcn.trainAndTestCross(modelPReal,modelPSyn,corrLPNBH,corrLPNBNH,corrLPNBH,\
    outputDir='predict/NBBH_Real_P_',sigmaL=[2],tTrain=tTrain,modeL=['0'])