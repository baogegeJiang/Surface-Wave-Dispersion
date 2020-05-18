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
        distance=np.arange(500,7200,600),srcSacNum=100,delta=1,layerN=20,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=300,\
        xcorrFuncL = [mathFunc.xcorrSimple],isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=11,dk=0.1,\
        fok='/k',order=0)
configTest=d.config(originName='models/ak135',srcSacDir=srcSacDir,\
        distance=np.arange(500,7200,600),srcSacNum=100,delta=1,layerN=28,\
        layerMode='prem',getMode = 'new',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=300,\
        xcorrFuncL = [mathFunc.xcorrSimple],isFlat=True,R=6371,flatM=-2,pog='p',calMode='gpdc',\
        T=T,threshold=0.02,expnt=11,dk=0.1,\
        fok='/k',order=0)

#config.genModel(N=1000,perD= 0.20,depthMul=2)
#configTest.genModel(N=1000,perD= 0.20,depthMul=2)
pL = []
f = fk.FK(orignExe=orignExe)
#fk.genSourceSacs(f,config.srcSacNum,config.delta,srcSacDir = srcSacDir,time=50)
#fk.genSourceSacs(f,config.srcSacNum,config.delta,srcSacDir = srcSacDirTest,time=50)

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
corrLLTest  = fkL(1000,FKCORRTest)
corrMatL = [np.array(corrL) for corrL in corrLL]
sio.savemat('mat/corrL_2048_1.mat',{'corrLL':corrMatL[0]})
corrMatL = 0
corrMatLTest = [np.array(corrLTest) for corrLTest in corrLLTest]
sio.savemat('mat/corrLTest_2048_1.mat',{'corrLL':corrMatLTest[0]})
corrMatLTest = 0
#corrMatTest = np.array(fkL(1000,FKCORRTest))
'''
for i in range(2):
    #with h5py.File('mat/corrLTest_%d_1.mat'%i, "w") as f:
    #    f['corrLL']=corrMatLTest[i]
    #with h5py.File('mat/corrL_%d_1.mat'%i, "w") as f:
    #    f['corrLL']=corrMatL[i]
    sio.savemat('mat/corrL_%d_1_1.mat'%i,{'corrLL':corrMatL[i][:150000]})
    sio.savemat('mat/corrL_%d_1_2.mat'%i,{'corrLL':corrMatL[i][150000:300000]})
    sio.savemat('mat/corrL_%d_1_3.mat'%i,{'corrLL':corrMatL[i][300000:]})
    sio.savemat('mat/corrLTest_%d_1_1.mat'%i,{'corrLL':corrMatLTest[i][:150000]})
    sio.savemat('mat/corrLTest_%d_1_2.mat'%i,{'corrLL':corrMatLTest[i][150000:300000]})
    sio.savemat('mat/corrLTest_%d_1_2.mat'%i,{'corrLL':corrMatLTest[i][300000:]})
    '''


corrMatL = []
corrMatTestL = []

corrMat = sio.loadmat('mat/corrL_2048_1.mat')['corrLL']
corrMatTest= sio.loadmat('mat/corrLTest_2048_1.mat')['corrLL']
#corrMat = np.concatenate(corrMatL)
#corrMatTest = np.concatenate(corrMatTestL)
fvGD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_g_0'%i,'file')for i in range(1000)}
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_new_p_0'%i,'file')for i in range(1000)}
fvGDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_g_0'%i,'file')for i in range(1000)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_new_p_0'%i,'file')for i in range(1000)}
fvGD['models/prem']= d.fv('models/prem_fv_flat_new_g_0','file')
fvPD['models/prem']= d.fv('models/prem_fv_flat_new_p_0','file')
fvGD['models/ak135']= d.fv('models/ak135_fv_flat_new_g_0','file')
fvPD['models/ak135']= d.fv('models/ak135_fv_flat_new_p_0','file')
for fv in fvGD:
    if fvGD[fv](1/120)>4:
        print(fv)
for fv in fvPD:
    if fvPD[fv](1/120)<4:
        print(fv)
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
    model.train(corrLTrain.x[:],corrLTrain.y[:],\
        xTest=corrLTest.x[:1000],yTest=corrLTest.y[:1000])
    iL=np.arange(0,1000,50)
    model.show(corrLTest.x[iL],corrLTest.y[iL],\
        time0L=corrLTest.t0L[0:1000:50],delta=1.0,T=tTrain,outputDir=outputDir)
    corrLTest.plotPickErro(model.predict(corrLTest.x[:]),tTrain,fileName=outputDir+'erro.jpg')

i = 0
#corrLP = d.corrL([ d.corr().setFromDict(tmpMat,isFile=True) for tmpMat in corrMat.reshape([-1])])
#corrLTestP = d.corrL([ d.corr().setFromDict(tmpMat,isFile=True) for tmpMat in corrMatTest.reshape([-1])[:20000]])
corrLP = d.corrL([ d.corr().setFromDict(tmpMat,isFile=True) for tmpMat in corrMat[0]])
corrLTestP = d.corrL([ d.corr().setFromDict(tmpMat,isFile=True) for tmpMat in corrMatTest[0][:20000]])
corrLP = d.corrL(config.modelCorr(1000))
corrLTestP = d.corrL(configTest.modelCorr(200))
corrLP = d.corrL( [ d.corr().setFromFile(corr.toMat())for corr in corrLP])
corrLTestP = d.corrL( [ d.corr().setFromFile(corr.toMat())for corr in corrLTestP])
#corrLP = d.corrL(corrLP)
#corrLTestP = d.corrL(corrLTestP)
corrLP = corrLL[0]
corrLTestP = corrLLTest[0]
#corrLP1 = d.corrL([ d.corr().setFromDict(tmpMat,isFile=False) for tmpMat in corrLL[1]])
#corrLTestP1 = d.corrL([ d.corr().setFromDict(tmpMat,isFile=False) for tmpMat in corrLLTest[1][:20000]])
#corrLP = d.corrL(corrLL[1])
#corrLTestP = d.corrL(corrLLTest[1])
corrLG     = corrLP.copy()
corrLTestG = corrLTestP.copy()
#corrLG1     = corrLP1.copy()
#corrLTestG1 = corrLTestP1.copy()

corrLP.getTimeDis(fvPD,tTrain,sigma=3,maxCount=1536,randD=30,byT=True)#,self1=corrLP1)
#,self1=corrLG1)
corrLTestP.getTimeDis(fvPDTest,tTrain,sigma=3,maxCount=1536,randD=30,byT=True)#,self1=corrLTestP1)
corrLG.getTimeDis(fvGD,tTrain,sigma=3,maxCount=1536,randD=30)
corrLTestG.getTimeDis(fvGDTest,tTrain,sigma=3,maxCount=1536,randD=30)#,self1=corrLTestG1)



modelP = fcn.model(channelList=[0])
modelG = fcn.model(channelList=[0])
trainAndTest(modelP,corrLP,corrLTestP,outputDir='predict/P_')
trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')
#trainAndTest(modelP,corrLP,corrLP,outputDir='predict/P_')
#trainAndTest(modelG,corrLG,corrLG,outputDir='predict/G_')
#trainAndTest(modelP,corrLP,corrLP,outputDir='predict/P_')
#trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')


#corrLTestG.plotPickErro(model.predict(corrLTestG.x[:]),tTrain)
'''
model.train(x[:100000],yG[:100000],xTest=xTest[:1000],yTest=yTestG[:1000])
iL=np.arange(0,1000,50)
model.show(xTest[iL],yTestG[iL],time0L=time0L[0:1000:50],delta=0.5,T=tTrain)

time0L=[tmpCorr.timeL[0] for tmpCorr in corrLTmp]
model.show(xTest[:20],yTestP[:20],time0L=time0L,delta=0.5,T=tTrain)
model.fit(x,y)

for corr in corrLTmp:
    plt.close()
    #corr = d.corr().setFromDict(tmpMat)
    corr.show(config.getDisp(),fvPDTest[corr.modelFile])
    plt.savefig('%s%d.jpg'%(disDir,i),dpi=300)
    print(i)
    i+=1
    plt.close()
    if i >20:
        break


outputDir ='predict/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

corrLTmp = random.sample(corrLTest,20)
time0L=[tmpCorr.timeL[0] for tmpCorr in corrLTmp]
x,y=d.getTimeDis(corrLTmp,fvPDTest,T=tTrain,sigma=2,maxCount=512)



K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.9)
'''

import seism
from obspy import UTCDateTime

stations = seism.StationList('stations/staLstNMV2SelectNew')
#stations.write('staLstAllNew')
#quakes   = seism.QuakeL('phaseLstVNM_20200305V1')
#quakes.write('phaseL')
quakes   = seism.QuakeL('phaseGlobal')
req ={\
'loc0':stations.loc0(),\
'maxDist':7200,\
'minDist':500,\
'time0':UTCDateTime(2014,1,1).timestamp,\
'time1':UTCDateTime(2017,1,1).timestamp\
}
quakes.select(req)
para ={\
'delta0' :1,
'freq'   :[0.8/1e3,0.8/2]
}
quakes.cutSac(stations,bTime=-10,eTime =2048,para=para,byRecord=False)



