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
#是否需要考虑展平变化的影响
orignExe='/home/jiangyr/program/fk/'
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
config=d.config(originName='models/prem',srcSacDir=srcSacDir,\
        distance=np.arange(400,1500,100),srcSacNum=100,delta=0.5,layerN=20,\
        layerMode='prem',getMode = 'norm',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=150,\
        xcorrFunc = mathFunc.xcorrSimple,isFlat=True,R=6371,flatM=-2,pog='p',calMode='fast',\
        T=np.array([0.5,1,5,10,20,30,50,80,100,150,200,250,300]),threshold=0.1,expnt=10,dk=0.1,\
        fok='/k')
configTest=d.config(originName='models/ak135',srcSacDir=srcSacDir,\
        distance=np.arange(400,1500,100),srcSacNum=100,delta=0.5,layerN=28,\
        layerMode='prem',getMode = 'norm',surfaceMode='PSV',nperseg=200,noverlap=196,halfDt=150,\
        xcorrFunc = mathFunc.xcorrSimple,isFlat=True,R=6371,flatM=-2,pog='p',calMode='fast',\
        T=np.array([0.5,1,5,10,20,30,50,80,100,150,200,250,300]),threshold=0.1,expnt=10,dk=0.1,\
        fok='/k')



f = fk.FK(orignExe=orignExe)
fk.genSourceSacs(f,config.srcSacNum,config.delta,srcSacDir = srcSacDir,time=50)

pL = []
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
corrMat = np.array(fkL(20,FKCORR))
sio.savemat('mat/corrL.mat',{'corrLL':corrMat})
FKCORRTest  = d.fkcorr(configTest)
corrMatTest = np.array(fkL(20,FKCORRTest))
sio.savemat('mat/corrLTest.mat',{'corrLL':corrMatTest})

fvGD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_norm_g'%i,'file')for i in range(1000)}
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_norm_p'%i,'file')for i in range(1000)}
fvGDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_norm_g'%i,'file')for i in range(100)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_norm_p'%i,'file')for i in range(100)}
disDir = 'disDir/'
if not os.path.exists(disDir):
    os.makedirs(disDir)

i = 0
corrL = [ d.corr().setFromDict(tmpMat) for tmpMat in corrMat]
model = fcn.model()
for i in range(1000):
    corrLTmp = random.sample(corrL,100)
    x,y=d.getTimeDis(corrLTmp,fvGD,T=np.array([5,10,20,30,50,80,100,150,200,250]),sigma=2,maxCount=512)
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

T = np.array([5,10,20,30,50,80,100,150,200,250])
outputDir ='predict/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
corrLTest = [ d.corr().setFromDict(tmpMat) for tmpMat in corrMatTest]
corrLTmp = random.sample(corrLTest,10)
time0L=[tmpCorr.timeL[0] for tmpCorr in corrLTmp]
x,y=d.getTimeDis(corrLTmp,fvPDTest,T=T,sigma=2,maxCount=512)
model.show(x[:10],y[:10],time0L=time0L,delta=0.5,T=T)