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
srcSacDirTest='/home/jiangyr/Surface-Wave-Dispersion/srcSacTest/'
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
corrMat = np.array(fkL(1000,FKCORR))
sio.savemat('mat/corrL.mat',{'corrLL':corrMat})
FKCORRTest  = d.fkcorr(configTest)
corrMatTest = np.array(fkL(1000,FKCORRTest))
sio.savemat('mat/corrLTest.mat',{'corrLL':corrMatTest})


corrMat = sio.loadmat('mat/corrL.mat')['corrLL']
corrMatTest = sio.loadmat('mat/corrLTest.mat')['corrLL']
fvGD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_norm_g'%i,'file')for i in range(1000)}
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_norm_p'%i,'file')for i in range(1000)}
fvGDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_norm_g'%i,'file')for i in range(1000)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_norm_p'%i,'file')for i in range(1000)}
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

i = 0
corrLP = d.corrL([ d.corr().setFromDict(tmpMat,isFile=True) for tmpMat in corrMat.reshape([-1])])
corrLTestP = d.corrL([ d.corr().setFromDict(tmpMat,isFile=True) for tmpMat in corrMatTest.reshape([-1])[:20000]])
corrLG     = corrLP.copy()
corrLTestG = corrLTestP.copy()
tTrain =np.array([5,10,20,30,50,80,100,150,200,250])
corrLP.getTimeDis(fvPD,tTrain,sigma=2,maxCount=512,randD=30)
corrLG.getTimeDis(fvGD,tTrain,sigma=2,maxCount=512,randD=30)
corrLTestP.getTimeDis(fvPDTest,tTrain,sigma=2,maxCount=512,randD=30)
corrLTestG.getTimeDis(fvGDTest,tTrain,sigma=2,maxCount=512,randD=30)




def trainAndTest(model,corrLTrain,corrLTest,outputDir='predict/'):
    tTrain =np.array([5,10,20,30,50,80,100,150,200,250])
    time0L = corrLTest.t0L
    model.train(corrLTrain.x[:],corrLTrain.y[:],\
        xTest=corrLTest.x[:1000],yTest=corrLTest.y[:1000])
    iL=np.arange(0,1000,50)
    model.show(corrLTest.x[iL],corrLTest.y[iL],\
        time0L=time0L[0:1000:50],delta=0.5,T=tTrain,outputDir=outputDir)
    corrLTestG.plotPickErro(model.predict(corrLTest.x[:]),tTrain,fileName=outputDir+'erro.jpg')
modelP = fcn.model()
modelG = fcn.model()
trainAndTest(modelP,corrLP,corrLTestP,outputDir='predict/P_')
trainAndTest(modelG,corrLG,corrLTestG,outputDir='predict/G_')


corrLTestG.plotPickErro(model.predict(corrLTestG.x[:]),tTrain)

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