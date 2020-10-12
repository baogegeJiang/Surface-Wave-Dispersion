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
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrAndDe],\
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


###handle CEA###
import dispersion as d
from imp import reload
import numpy as np
import os
import mathFunc
import fcn
import seism
#是否需要考虑展平变化的影响
orignExe='/home/jiangyr/program/fk/'
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
srcSacDirTest='/home/jiangyr/Surface-Wave-Dispersion/srcSacTest/'
T=np.array([0.5,1,2,5,8,10,15,20,25,30,40,50,60,70,80,100,125,150,175,200,225,250,275,300])


path='/media/commonMount/data4/CEAdata/2018/'
stationsCEA = seism.StationList('stations/CEA.sta_sel')
'''
#select quakes
stations = seism.StationList('stations/CEA.sta_sel')
quakes   = seism.QuakeL('phaseGlobal')
req ={\
'loc0':stations.loc0(),\
'maxDist':10000,\
'minDist':500,\
'time0':obspy.UTCDateTime(2009,1,1).timestamp,\
'time1':obspy.UTCDateTime(2013,1,1).timestamp\
}
quakes.select(req)
quakes.write('phaseLCEAV1_more')

'''
#select more
'''
quakes   = seism.QuakeL('phaseL_200909_201109_4')
req ={\
'loc0':stations.loc0(),\
'maxDist':10000,\
'minDist':500,\
'time0':obspy.UTCDateTime(2009,9,17).timestamp,\
'time1':obspy.UTCDateTime(2011,8,5).timestamp\
}
quakes.select(req)
for i in range(len(quakes)-1,-1,-1):
    quake = quakes[i]
    delta, dk, az = quake.calDelta(stations.loc0()[0],\
        stations.loc0()[1])
    if delta>20 and quake['ml']<4.5:
        quakes.pop(i)
        continue
    if delta>40 and quake['ml']<5:
        quakes.pop(i)

quakes.write('phaseLCEAV2')
'''
#cut sacs for quakes
quakesCEA   = seism.QuakeL('phaseLCEAV2')
stationsCEA = seism.StationList('stations/CEA.sta_sel')

'''
#cut sacs for quake
para={\
'delta0' :1,
'freq'   :[-1,-1],#[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e10,
}

quakes.cutSac(stations,bTime=-10,eTime =4096,\
    para=para,byRecord=False,isSkip=True)
for quake in quakes[:]:
    quake.getSacFiles(stations,isRead=True,remove_resp=False,\
        isPlot=False,isSave=True,para={'freq':[0.8/3e2,0.8/2]},\
        isSkip=True)
'''

para={'freq'      :[1/300],'filterName':'highpass'}
para={'freq'      :[1/300,1/6],'filterName':'bandpass'}
#para={'freq'      :[-1,-1],'filterName':'bandpass'}
import dispersion as d
import fk
from imp import reload
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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

para={'freq'      :[1/6],'filterName':'lowpass'}
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=110*10,maxDist=110*170,minDDist=200,\
        maxDDist=10000,para=para,isFromO=True,removeP=True)

quakes = seism.QuakeL('phaseLPickCEA')

stationsCEA = seism.StationList('stations/CEA.sta_sel')
stationsCEA.set('oRemove',True)#True
stationsNE = seism.StationList('stations/NEsta_all.locSensorDas')
stationsNE.set('oRemove',False)
stations = stationsCEA#+stationsNE
stations.getInventory()
random.shuffle(quakes)
#stations.getSensorDas()
#fvNED.update(fvDAvarageCEA)
#fvDAvarageNE = config.loadNEFV(stationsNE)
fvDAvarageCEA = d.config().loadNEFV(stationsCEA,fvDir='models/ayu/Pairs_avgpvt/')
fvNED,quakes0 = d.config().loadQuakeNEFV(stations,quakeFvDir='models/ayu/Pairs_pvt/')
quakes = quakes0
fvDAvarage = {}
d.replaceByAv(fvNED,fvDAvarageCEA)
fvDAvarage.update(fvNED)
#fvDAvarage.update(fvDAvarageNE)
fvDAvarage['models/prem']=d.fv('models/prem_fv_flat_new_p_0','file')
SNR = 8#12
'''
corrLQuakeP0=config.quakeCorr(quakes,stations,\
    byRecord=False,remove_resp=True,minSNR=SNR,isLoadFv=True,\
    fvD=fvDAvarage,isByQuake=True,para={\
    'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
        'output':'VEL','freq':[1/300, 1/5],'filterName':'bandpass',\
        'corners':4,'toDisp':False,'zerophase':True,'maxA':1e15})
'''
corrLQuakeP0=config.quakeCorr(quakes,stations,\
    byRecord=False,remove_resp=True,minSNR=SNR,isLoadFv=False,\
    fvD=fvDAvarage,isByQuake=True,para={\
    'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
        'output':'VEL','freq':[1/300, 1/5],'filterName':'bandpass',\
        'corners':4,'toDisp':False,'zerophase':True,'maxA':1e15})
corrLQuakeP0 = d.corrL(corrLQuakeP0,maxCount=4096*3)
corrLQuakeP = d.corrL(corrLQuakeP0,maxCount=4096*3,fvD=fvNED)
#corrLQuakeP = d.corrL(corrLQuakeP)
#考虑取出其中的一些地震，避免数据泄露
fvL = [key for key in fvDAvarageCEA]
random.shuffle(fvL)
fvTrain = fvL[100:]
fvTest = fvL[50:100]
fvVaild = fvL[:50]
specThreshold = 0.0
corrLQuakePTrain = d.corrL(corrLQuakeP,specThreshold=specThreshold,fvD=fvTrain)
corrLQuakePTest  = d.corrL(corrLQuakeP,specThreshold=specThreshold,fvD=fvTest)
#corrLQuakePTrain = d.corrL(corrLQuakePCEA[:-1000])
corrLQuakePValid = d.corrL(corrLQuakeP,specThreshold=specThreshold,fvD=fvVaild)
#corrLQuakePTest  = d.corrL(corrLQuakePNE)
#random.shuffle(corrLQuakePTrain)
#random.shuffle(corrLQuakePValid)
#random.shuffle(corrLQuakePTest)
tTrain = (16**np.arange(0,1.000001,1/49))*10
corrLQuakePTrain.setTimeDis(fvDAvarage,tTrain,sigma=1.5,maxCount=4096*3,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
set2One=True,move2Int=False)
corrLQuakePValid.setTimeDis(fvDAvarage,tTrain,sigma=1.5,maxCount=4096*3,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
set2One=True,move2Int=False)
corrLQuakePTest.setTimeDis(fvDAvarage,tTrain,sigma=1.5,maxCount=4096*3,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=False,\
set2One=True,move2Int=False)#0.05
modelP = fcn.model(channelList=[0,2,3])
fcn.trainAndTest(modelP,corrLQuakePTrain,corrLQuakePValid,corrLQuakePTest,\
    outputDir='predict/0928byQuake_dense+++_1-300_1-5_allllRecord_w3_L7_count05_SNR8/CEA_P_',\
    sigmaL=[1.5],tTrain=tTrain,perN=200,count0=5,w0=3)#w0=2

corrLQuakeP0 = d.corrL(corrLQuakeP0,maxCount=4096*3)

corrLQuakeP0.setTimeDis(fvDAvarage,tTrain,sigma=1.5,maxCount=4096*3,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=True,\
set2One=True,move2Int=False,modelNameO='models/prem')


stationsNE = seism.StationList('stations/NEsta_all.locSensorDas')
stationsNE.set('oRemove',False)
stations = stationsNE
stations.getInventory()

quakes  = seism.QuakeL('phaseLCEAV2_more')
corrLQuakeP0=config.quakeCorr(quakes,stations,\
    byRecord=False,remove_resp=True,minSNR=SNR,isLoadFv=False,\
    fvD=fvDAvarage,isByQuake=True,para={\
    'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
        'output':'DISP','freq':[1/5],'filterName':'lowpass',\
        'corners':4,'toDisp':False,'zerophase':True,'maxA':1e15})
corrLQuakeP0 = d.corrL(corrLQuakeP0,maxCount=4096*3)

corrLQuakeP0.setTimeDis(fvDAvarage,tTrain,sigma=1.5,maxCount=4096*3,\
byT=False,noiseMul=0.0,byA=True,rThreshold=0.0,byAverage=True,\
set2One=True,move2Int=False,modelNameO='models/prem')

resDir = 'predict/v23ByQuake_NE/'
corrLQuakeP0.getAndSave(modelP,'%s/CEA_P_'%resDir,stations\
    ,isPlot=True,isLimit=True,isSimple=False,D=0.2)
resDir = 'predict/v23ByQuake_NE/'
fvNEDGet,quakesGet = d.config().loadQuakeNEFV(stations,quakeFvDir=resDir)
fvDAvarageCEA = d.config().loadNEFV(stationsCEA,fvDir='models/ayu/Pairs_avgpvt/')
fvM=d.fvD2fvM(fvNEDGet,isDouble=True)
fvAv = d.fvM2Av(fvM)
fvMean = d.averageFVL([fvAv[key] for key in fvAv])
fvPrem=d.fv('models/prem_fv_flat_new_p_0','file')
#d.plotFVM(fvM,fvD=fvAv,resDir='%s/pairRes/'%resDir,isDouble=True)

d.plotFVL(fvAv,fvRef=fvPrem,filename='%s/pairRes/pair.jpg'%resDir)
for fv in fvAv:
    fvAv[fv].qc()

d.qcFvD(fvAv)
len(fvAv)
d.plotFVL(fvAv,fvRef=fvPrem,filename='%s/pairRes/paris_qc.jpg'%resDir)
d.compareFvD(fvAv,fvDAvarageCEA,resDir='%s/pairRes/paris_qc/'%resDir)

d.qcFvD(fvAv)
f = 1/tTrain


indexL,vL = d.fvD2fvL(fvAv,stations,f)
import DSur
surPara={ 'nxyz':[38,42,18], 'lalo':[29,98],\
         'dlalo':[0.25,0.25], 'maxN':[43],\
        'kmaxRc':50,'rcPerid':tTrain.tolist(),'threshold':0.05,\
        'vnn':[0,100,50],'maxIT':15}
DSConfig = DSur.config(para=surPara,z=[0.00,5.00,10.00,15.00,20.00,25.00,30.00,35.00,\
    40.00,45.00,50.00,55.00,60.00,70.00,80.00,90.00,100.00,120.00])
DS = DSur.DS(config=DSConfig)
DS.test(vL,indexL,stations)
DSModel = DS.loadRes()
DSModel.plotByZ()
############
indexL,vL = d.fvD2fvL(fvDAvarageCEA,stations,f)

tSur = (16**np.arange(0,1.000001,1/29))*10
indexL,vL = d.fvD2fvL(fvAv,stations,1/tSur)

z=[0,10,20,30,40,60,80,100,125,150,180,200,250,300,350]
z=[0]+(35**np.arange(0,1,1/11)*10).tolist()
import DSur
surPara={ 'nxyz':[36,54,len(z)], 'lalo':[56,109],\
         'dlalo':[0.5,0.5], 'maxN':43,\
        'kmaxRc':len(tSur),'rcPerid':tSur.tolist(),'threshold':0.01\
        ,'maxIT':6}
DSConfig = DSur.config(para=surPara,z=z)
DS = DSur.DS(config=DSConfig,runPath='DS/NE160/')
DS.test(vL,indexL,stations)
DS.loadRes()
DS.plotByZ()

'''
handle some preparation
respDirs = ['resp/RESP/','resp/RESP_old']
staFiles  = ['stations/CEA.sta','stations/CEA_old.sta']
for i in range(2):
    respDir = respDirs[i]
    staFile = staFiles[i]
    staL = seism.StationList([])
    staL.inD['keysIn']='net sta la lo dep compBase sensorName dasName nameMode'
    for netDir in os.listdir(path):
        net = os.path.basename(netDir)
        for staDir in os.listdir(path+'/'+netDir):
            sta = os.path.basename(staDir)
            fileL = glob(path+netDir+'/'+staDir+'/2010/2*0/*Z.sac')
            if len(fileL)==0:
                continue
            else:
                file = fileL[0]
            sac = obspy.read(file)[0]
            la = sac.stats['sac'][ 'stla']
            lo = sac.stats['sac'][ 'stlo']
            el = sac.stats['sac'][ 'stel']
            station = seism.Station()
            station['net'] = net
            station['sta'] = sta
            station['la'] = la
            station['lo'] = lo
            station['dep'] = el
            station['compBase']=file.split('.')[-2][:-1]
            station['sensorName']='%s/RESP.%s.%s.00.BHZ'%(respDir,net,sta)
            station['dasName']='130s'
            station['nameMode']='CEA'
            staL.append(station)
            print(station)
    staL.write(staFile)
        #resp/YP/RESP.YP.NE11..BHZ 130S NECE
        #RESP.HE.CXT.00.BHZ

stations = seism.StationList('stations/CEA.sta')
fvNEDAvarage = config.loadNEFV(stations,fvDir='models/Curves/')
staNames = []
for key in fvNEDAvarage:
    stas = key.split('_')
    sta0 = stas[0]
    sta1 = stas[1]
    if sta0 not in staNames:
        staNames.append(sta0)
    if sta1 not in staNames:
        staNames.append(sta1)

for station in stations:
    key = station['net'] + '.' + station['sta']
    if key  not in staNames:
        stations.remove(station)

stations.write('stations/CEA.sta_sel')
'''

'''
是否需要质量控制？
1800*1.5=2700
差距大的扔掉
然后重新计算均值
可以比单个的更加接近
list 记顺序
dictionary 存内容

现根据path 算每条路径上面波速度对结果的影响
再算对应地点的三维结构对面波速度的影响

实际数据训练，模拟数据测试？
模拟数据训练，实际数据测试？
只训练某些部分

调整训练参数 如sigma

sta0 sta1 period
多做几次循环即可

更严格地控制数据质量

是否用将面波与P波比
模型大小，容纳程度 

与多少个台有关系


多大震中距的可用
用速度还是位移？

增加地震数量？
可训练而不可做要求？
phaseLCEAV1_more 后255个无地震
时候需要滤波？
预处理操作

***寻找大周期误差的来源（与标注数据的来源有关?）

重新对内蒙台站进行去仪器响应？
是否应该加入噪声的频散曲线提取？

找到原始的手挑数据，自己筛选

在发表正式版的时候，应该注意尽量精简
让读者自己提供cut好的文件
是否需要考虑噪音的自动提取
但看起来似乎不太需要，因为噪音对的数目本来就少
和自动拾取结果比较？
扩大拾取频率范围
频谱相似要求？
是否需要去除仪器响应？
研究初始化问题，如何保证收敛
利用姚老师的方法
台站距离
看残差大小

是否可以继续上采样，寻找可能的超分辨率？
虽然加入了训练系统，但是部分频段仍然未被标注，也可以用于拾取
如何从predicate 结果中选取有用的
fit 的范式, 可以提高精度吗
峰的宽度要和数据的误差匹配
最好不好要有同名或者换过台的台阵当两个台阵处理
stationTrain
stationTest？
和prem标准模型比较比较
需要很大的感受野，所以deepSupersion对结果影响不大
不一定需要DS
是否只用输入xx？
看能否降低层数
尝试不同的w sigma
似乎更低的sigma管用
是否需要把时间对应到具体的刻度上？
移动到整数的点上
是否需要将其调整成整数
是否限制最大时差？
详细考虑如何控制数据质量
某个波段足够相似才行？
是否应该限制相似度
'''
'''
stations = seism.StationList('stations/CEA.sta_sel')
quakes   = seism.QuakeL('phaseGlobal')
req ={\
'loc0':stations.loc0(),\
'maxDist':1e10,\
'minDist':-10,\
'time0':obspy.UTCDateTime(2009,1,1).timestamp,\
'time1':obspy.UTCDateTime(2013,1,1).timestamp\
}
quakes.select(req)
quakes.write('phaseLCEAV2_more')
'''

#cut sacs for quake
import seism
#import dispersion as d 
#quakes  = seism.QuakeL('phaseLCEAV2_more')
#quakes  = seism.QuakeL('phaseLPick')
#stations = seism.StationList('stations/CEA.sta_sel')
#fvNED,quakes0 = d.config().loadQuakeNEFV(stations,quakeFvDir='models/ayu/Pairs_pvt/')
#quakes0=seism.QuakeL(quakes0)
#quakes0.write('phaseLPickCEA')
#+seism.StationList('stations/NEsta_all.locSensorDas')
#quakes  = seism.QuakeL('phaseLPickCEA')
'''
popL = []
for i in range(len(stations)):
    station = stations[i]
    if station['compBase'] =='SH':
        popL.append(i)
for pop in popL[-1::-1]:
    stations.pop(pop)

numL = [[] for sta in stations]
for i in range(len(stations)):
    for j in range(len(stations)):
        if stations[i].dist(stations[j])< 50:
            numL[i].append(j)
moreL = []
for i in range(len(stations)):
    if len(numL[i])>3:
        moreL.append(i)
'''
para={\
'delta0' :1,
'freq'   :[-1,-1],#[0.8/3e2,0.8/2],
'corners':4,
'maxA':1e19,
}

quakes.cutSac(stations,bTime=-1500,eTime =12300,\
    para=para,byRecord=False,isSkip=True)

#stations = seism.StationList('stations/NEsta_all.locSensorDas')
stations = seism.StationList('stations/CEA.sta_sel')
#stations = seism.StationList('stations/CEA.sta_know_few')
stations.getInventory()
quakes  = seism.QuakeL('phaseLCEAV2_more')
quakes  = seism.QuakeL('phaseLPickCEA')
quakes.sort()
for quake in quakes[2::3]:
    print(quake)
    a=quake.getSacFiles(stations,isRead=True,remove_resp=True,\
        isPlot=False,isSave=True,para={'freq':[-1,-1],\
        'pre_filt': (1/400, 1/300, 1/2, 1/1.5),\
        'output':'VEL'},isSkip=True)
stations = seism.StationList('stations/CEA.sta_sel')
quakes   = seism.QuakeL('phaseLNE')
stations = seism.StationList('stations/NEsta_all.locSensorDas')
fvNEDAvarage = config.loadNEFV(stations)

'''
self.dtype = self.getDtype(maxCount)
        self.xx    = xx.astype(np.complex64)
        self.timeL = timeL.astype(np.float32)
        self.dDis  = dDis
        self.fs    = fs
        self.az    = az
        self.dura  = dura
        self.M     = M
        self.dis   = dis
        self.dep   = dep
        self.modelFile=modelFile
        self.name0 = name0
        self.name1 = name1
        self.srcSac= srcSac
        self.x0 = x1.astype(np.float32)
        self.x0 = x1.astype(np.float32)
        self.quakeName = quakeName
        '''
reload(d)
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(500,10000,300),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,\
        noverlap=196,halfDt=300,xcorrFuncL = [mathFunc.xcorrFrom0],\
        isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=12,dk=0.05,\
        fok='/k',order=0,minSNR=10,isCut=False,\
        minDist=600,maxDist=10000,minDDist=300,\
        maxDDist=3000,para=para,isFromO=True,removeP=True)
corrLQuakePCEA = d.corrL(config.quakeCorr(quakesCEA[:100],stationsCEA,\
    byRecord=False,remove_resp=False,minSNR=8,isLoadFv=True,\
    fvD=fvDAvarage,isByQuake=False))
cspec = []
for corr in corrLQuakePCEA:
    cspec.append(corr.compareSpec(N=40))

##
t = np.arange(500)
w = 1/20*np.pi*2

x0 = np.sin(w*t)
x1 = np.sin(w*(t-0.45))

X0 = np.fft.fft(x0)
X1 = np.fft.fft(x1)

phase = np.imag(X1[25]/X0[25])/w

for corr in corrLQuakeP :
    if corr.xx.std()==0 :
        print(corr.name0)
    if corr.x0.std()==0 :
        print(corr.name0)
    if corr.x1.std()==0 :
        print(corr.name0)

for corr in corrLQuakeP :
    for x in [corr.xx, corr.x1, corr.x0]:
        if np.isnan(x).sum()>0:
            print(1)

N = len(corrLQuakeP)
loopN = int(N/1000)+1
for i in range(loopN):
    print(i)
    x,y,t=corrLQuakeP(np.arange(1000*i,min(1000*(i+1),N)))
    S =x[:,:,:,[0,2,3]].std(axis=1)
    if (S==0).sum()>0:
        print(np.where(S==0))

N = len(corrLQuakePTest)
loopN = int(N/1000)+1
for i in range(loopN):
    print(i)
    x,y,t=corrLQuakePTest(np.arange(1000*i,min(1000*(i+1),N)))
    S =x[:,:,:,[0,2,3]].std(axis=1)
    if (S==0).sum()>0:
        print(np.where(S==0))

import seism
from glob import glob
import os 
import obspy
stations = seism.StationList('stations/CEA.sta')
YMDL=glob('/media/jiangyr/CEA/CEA_1s/????????/')

for YMD in YMDL[43:]:
    strT=YMD.split('/')[-2]
    time = obspy.UTCDateTime(strT)
    YJ   = time.strftime('%Y/%j/')
    for station in stations:
        net = station['net']
        sta = station['sta']
        tmpDistDir = '/net/CEA/CEA1s/CEA/%s/%s/%s/'%(net,sta,YJ)
        if not os.path.exists(tmpDistDir):
            mkdir = 'echo a314349798 | sudo mkdir -p %s'%(tmpDistDir)
            os.system(mkdir)
        if len(glob('%s/%s*%s*'%(YMD,sta,net)))>0:
            cmd = 'echo a314349798 | sudo cp -n %s/%s*%s* %s/'%(YMD,sta,net,tmpDistDir)
            os.system(cmd)
            print(cmd)
from glob import glob
from obspy import UTCDateTime
i =0 
for quake in quakes0:
    resDir = '%s/%s/*BHZ'%('eventSac',quake.name('_'))
    if len(glob(resDir))==0:
        i+=1
        print(UTCDateTime(quake['time']),i)
        #if os.path.exists(os.path.dirname(resDir)):
        #    os.system('rm -r %s'%os.path.dirname(resDir))

import run

R  = run.run()
R.calResOneByOne()
R.loadCorr()
R.train()

disL = []
N = len(stations)
for i in range(N):
    for j in range(i):
        disL.append(stations[i].dist(stations[j]))


distL=[]
for corr in R.corrL:
    distL.append(corr.dDis)
最大1800
最小200