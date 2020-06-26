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
        minDist=600,maxDist=10000,minDDist=200,\
        maxDDist=3000,para=para,isFromO=True,removeP=True)
configTest=d.config(originName='models/ak135New',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=39,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=200,maxDDist=3000,para=para,\
        isFromO=True,removeP=True)

configNoQ=d.config(originName='models/noQ',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=1000,maxDist=1e8,minDDist=200,maxDDist=3000,para=para,\
        isFromO=True,removeP=True)


config.genModel(N=1000,perD= 0.30,depthMul=2)
configTest.genModel(N=1000,perD= 0.30,depthMul=2)
#configNoQ.genModel(N=1000,perD= 0.30,depthMul=2)



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
FKCORRNoQ  = d.fkcorr(configNoQ)
corrLLNoQ  = fkL(20,FKCORRNoQ)

corrMatL = []
corrMatTestL = []


fvGD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_g_0'%i,'file')for i in range(1000)}
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_p_0'%i,'file')for i in range(1000)}
fvGDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_g_0'%i,'file')for i in range(1000)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_p_0'%i,'file')for i in range(1000)}
fvGD['models/prem']= d.fv('models/prem_fv_flat_new_g_0','file')
fvPD['models/prem']= d.fv('models/prem_fv_flat_new_p_0','file')
fvGD['models/ak135']= d.fv('models/ak135_fv_flat_new_g_0','file')
fvPDTest['models/ak135']= d.fv('models/ak135_fv_flat_new_p_0','file')
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
tTrain = (10**np.arange(0,1.000001,1/29))*16
'''
def trainAndTest(model,corrLTrain,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,3,2,1.5]):
    '''
    #依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    w0 = model.config.lossFunc.w
    for sigma in sigmaL:
        model.config.lossFunc.w = 10*4/sigma
        corrLTrain.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLTest.iL=np.array([])
        model.compile(loss=model.config.lossFunc, optimizer='Nadam')
        xTest, yTest, tTest =corrLTest(np.arange(3000,6000))
        model.trainByXYT(corrLTrain,xTest=xTest,yTest=yTest)
        

    xTest, yTest, tTest =corrLTest(np.arange(3000))
    corrLTest.plotPickErro(model.predict(xTest),tTrain,\
    fileName=outputDir+'erro.jpg')
    iL=np.arange(0,1000,50)
    model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir)
'''
    

i = 0
stationsN = seism.StationList('stations/staLstNMV2SelectNewSensorDasCheck') + seism.StationList('stations/NEsta_all.locSensorDas')
stationsN.getInventory()
noises=seism.QuakeL('noiseL') + seism.QuakeL('noiseLNE')
n = config.getNoise(noises,stationsN,mul=3,para=para,\
    byRecord=False,remove_resp=True)
n.mul = 4
corrLP = d.corrL(config.modelCorr(1000,noises=n,randDrop=0.3,minSNR=0.1))
corrLTestP = d.corrL(configTest.modelCorr(100,noises=n,randDrop=0.2,minSNR=0.1))
#corrLG     = corrLP.copy()
#corrLTestG = corrLTestP.copy()


#corrLP = d.corrL(corrLP)
#corrLTestP = d.corrL(corrLTestP)
corrLP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0)
corrLTestP.setTimeDis(fvPDTest,tTrain,sigma=4,\
maxCount=4096,byT=False,noiseMul=0.0)
'''
corrLG.setTimeDis(fvGD,tTrain,sigma=6,maxCount=4096,\
noiseMul=0.0)
corrLTestG.setTimeDis(fvGDTest,tTrain,sigma=6,\
maxCount=4096,noiseMul=0.0)
'''

#modelP = fcn.model(channelList=[0,2,3])
#modelG = fcn.model(channelList=[0,2,3])
modelP = fcn.model(channelList=[0])
fcn.trainAndTest(modelP,corrLP,corrLTestP,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)

#fcn.trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')

quakes   = seism.QuakeL('phaseL')

corrLQuakeP = d.corrL(config.quakeCorr(quakes[80:100],stations,\
    False,remove_resp=True,para=para,minSNR=40))
corrLQuaketP = d.corrL(corrLQuakeP)
corrLQuakeP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,byT=False)
xQuake, yQuake, tQuake =corrLQuakeP(np.arange(0,40000,1000))
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


import seism
from obspy import UTCDateTime


#stations.write('staLstAllNew')
#quakes   = seism.QuakeL('phaseLstVNM_20200305V1')
#quakes.write('phaseL')
stations = seism.StationList('stations/NEsta_all.loc')
stations.getSensorDas()
stations.write('stations/NEsta_all.locSensorDas',\
    'net sta compBase la lo dep sensorName dasName sensorNum')
quakes   = seism.QuakeL('phaseGlobal')
stations = seism.StationList('stations/NEsta_all.locSensorDas')
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

quakes[:1000].cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False,isSkip=True)
for quake in quakes[:]:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]},isSkip=True)

quakes   = seism.QuakeL('phaseLPick')
stations.getSensorDas()
stations.getInventory()
para={\
'delta0' :1,
'freq'   :[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e10,
}
quakes[:].cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False,isSkip=True)
for quake in quakes[:]:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]},isSkip=True)


quakes   = seism.QuakeL('phaseLNE')
noises = quakes.copy()
for noise in noises:
    noise['time']-=5000

noises.write('noiseLNE')
noises=seism.QuakeL('noiseLNE')
stations = seism.StationList('stations/NEsta_all.locSensorDas')
stations.getInventory()
noises[0:100:5].cutSac(stations,bTime=-10,eTime =4096,para=para,byRecord=False)
#noises[:10].getSacFiles(stations)
#quakes[:10].getSacFiles(stations)
for quake in noises:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]})

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
quakes[-300:].cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False)
quakes   = seism.QuakeL('phaseL')
noises = quakes.copy()
for noise in noises:
    noise['time']-=5000
noises.write('noiseL')
noises=seism.QuakeL('noiseL')
noises[0:100:5].cutSac(stations,bTime=-10,eTime =4096,para=para,byRecord=False)
#noises[:10].getSacFiles(stations)
#quakes[:10].getSacFiles(stations)
for quake in quakes:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]})
for quake in noises:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]})
'''
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

configNoQ=d.config(originName='models/noQ',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.1,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=1000,maxDist=1e8,minDDist=200,maxDDist=3000,para=para,\
        isFromO=True,removeP=True)


#config.genModel(N=1000,perD= 0.30,depthMul=2)
#configTest.genModel(N=1000,perD= 0.30,depthMul=2)
#configNoQ.genModel(N=1000,perD= 0.30,depthMul=2)



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
FKCORRNoQ  = d.fkcorr(configNoQ)
corrLLNoQ  = fkL(20,FKCORRNoQ)

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
tTrain = (10**np.arange(0,1.000001,1/29))*16
'''
def trainAndTest(model,corrLTrain,corrLTest,outputDir='predict/',tTrain=tTrain,\
    sigmaL=[4,3,2,1.5]):
    '''
    #依次提高精度要求，加大到时附近权重，以在保证收敛的同时逐步提高精度
    '''
    #xTrain, yTrain, timeTrain =corrLTrain(np.arange(0,20000))
    #model.show(xTrain,yTrain,time0L=timeTrain ,delta=1.0,T=tTrain,outputDir=outputDir+'_train')
    w0 = model.config.lossFunc.w
    for sigma in sigmaL:
        model.config.lossFunc.w = 10*4/sigma
        corrLTrain.timeDisKwarg['sigma']=sigma
        corrLTest.timeDisKwarg['sigma']=sigma
        corrLTest.iL=np.array([])
        model.compile(loss=model.config.lossFunc, optimizer='Nadam')
        xTest, yTest, tTest =corrLTest(np.arange(3000,6000))
        model.trainByXYT(corrLTrain,xTest=xTest,yTest=yTest)
        

    xTest, yTest, tTest =corrLTest(np.arange(3000))
    corrLTest.plotPickErro(model.predict(xTest),tTrain,\
    fileName=outputDir+'erro.jpg')
    iL=np.arange(0,1000,50)
    model.show(xTest[iL],yTest[iL],time0L=tTest[iL],delta=1.0,\
    T=tTrain,outputDir=outputDir)
'''
    

i = 0
stations = seism.StationList('stations/staLstNMV2SelectNewSensorDasCheck')
stations.getInventory()
noises=seism.QuakeL('noiseL')
n = config.getNoise(noises,stations,mul=3,para=para,\
    byRecord=False,remove_resp=True)
n.mul = 2
corrLP = d.corrL(config.modelCorr(1000,noises=n,randDrop=0.3,minSNR=0.1))
corrLTestP = d.corrL(configTest.modelCorr(100,noises=n,randDrop=0.2,minSNR=0.1))
#corrLG     = corrLP.copy()
#corrLTestG = corrLTestP.copy()


#corrLP = d.corrL(corrLP)
#corrLTestP = d.corrL(corrLTestP)
corrLP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,\
byT=False,noiseMul=0.0)
corrLTestP.setTimeDis(fvPDTest,tTrain,sigma=4,\
maxCount=4096,byT=False,noiseMul=0.0)
'''
corrLG.setTimeDis(fvGD,tTrain,sigma=6,maxCount=4096,\
noiseMul=0.0)
corrLTestG.setTimeDis(fvGDTest,tTrain,sigma=6,\
maxCount=4096,noiseMul=0.0)
'''

#modelP = fcn.model(channelList=[0,2,3])
#modelG = fcn.model(channelList=[0,2,3])
modelP = fcn.model(channelList=[0])
fcn.trainAndTest(modelP,corrLP,corrLTestP,outputDir='predict/P_',sigmaL=[4,2],tTrain=tTrain)
#fcn.trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')

quakes   = seism.QuakeL('phaseL')

corrLQuakeP = d.corrL(config.quakeCorr(quakes[80:100],stations,\
    False,remove_resp=True,para=para,minSNR=40))
corrLQuaketP = d.corrL(corrLQuakeP)
corrLQuakeP.setTimeDis(fvPD,tTrain,sigma=4,maxCount=4096,byT=False)
xQuake, yQuake, tQuake =corrLQuakeP(np.arange(0,40000,1000))
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


import seism
from obspy import UTCDateTime


#stations.write('staLstAllNew')
#quakes   = seism.QuakeL('phaseLstVNM_20200305V1')
#quakes.write('phaseL')
stations = seism.StationList('stations/NEsta_all.loc')
stations.getSensorDas()
stations.write('stations/NEsta_all.locSensorDas',\
    'net sta compBase la lo dep sensorName dasName sensorNum')
quakes   = seism.QuakeL('phaseGlobal')
stations = seism.StationList('stations/NEsta_all.locSensorDas')
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
stations.getInventory()
for quake in quakes[:50]:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]})


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
quakes[-300:].cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False)
quakes   = seism.QuakeL('phaseL')
noises = quakes.copy()
for noise in noises:
    noise['time']-=5000
noises.write('noiseL')
noises=seism.QuakeL('noiseL')
noises[0:100:5].cutSac(stations,bTime=-10,eTime =4096,para=para,byRecord=False)
#noises[:10].getSacFiles(stations)
#quakes[:10].getSacFiles(stations)
for quake in quakes:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]})
for quake in noises:
    quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq'      :[0.8/3e2,0.8/2]})
'''