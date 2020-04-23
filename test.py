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
m = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem')
m.covert2Fk(0)
m.covert2Fk(1)
F,v =m.calDispersion(order=0,calMode='fast',threshold=0.1,T=3*10**np.arange(0,2.1,0.2),pog='g')
F,v =m.calDispersion(order=0,calMode='fast',threshold=0.1,T=3*10**np.arange(0,2.1,1),pog='g')
plt.axes(xscale = "log");plt.plot(1/F,v);plt.show()
orignExe='/home/jiangyr/program/fk/'
f = fk.FK(orignExe=orignExe)
sacsL,sacNamesL = f.test(distance=[1000,1100],modelFile='prem',fok='/f',dt=1,depth=20,\
    expnt=9,dura=20,dk=0.1,srcSac='../sourceSac.sac')
disp = d.disp(nperseg=100,noverlap=98,fs=1,halfDt=150,xcorrFunc = mathFunc.xcorrSimple)
corr=disp.testSac(sacsL[0][1][0],sacsL[0][0][0],fTheor=F,vTheor=v);plt.show()
#d.genModel(modelFile = 'prem',N=1000,perD = 0.10,depthMul=2)
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'
#d.genModel(modelFile = 'ak135',N=1000,perD = 0.10,depthMul=2)




srcSacNum = 100
delta=0.5
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
f = fk.FK(orignExe=orignExe)
fk.genSourceSacs(f,srcSacNum,delta,srcSacDir = srcSacDir,time=50)
D = d.disp(nperseg=200,noverlap=196,fs=1/delta,halfDt=150,xcorrFunc = mathFunc.xcorrSimple)
distance=np.arange(400,1500,100)


f = fk.FK(orignExe=orignExe)
pL = []
pN = 5
for i in range(pN):
    pL.append(  multiprocessing.Process(target=d.calFv, args=(range(i,1000,pN), 'models/prem',20,'p')))
    pL.append(  multiprocessing.Process(target=d.calFv, args=(range(i,1000,pN), 'models/prem',20,'g')))
    pL.append(  multiprocessing.Process(target=d.calFv, args=(range(i,1000,pN), 'models/ak135',28,'p')))
    pL.append(  multiprocessing.Process(target=d.calFv, args=(range(i,1000,pN), 'models/ak135',28,'g')))
    #(range(i,100,pN), 'models/ak135',28,'p')
    #(range(i,100,pN), 'models/prem',20,'p')

for p in pL:
    p.start()

for p in pL:
    p.join()

corrMat = d.multFK(2,d.singleFk,2,D,'models/prem',srcSacDir,distance,srcSacNum,delta,\
    orignExe=orignExe)
sio.savemat('corrL.mat',{'corrLL':corrMat})

corrMatTest = d.multFK(20,d.singleFk,20,D,'models/ak135',srcSacDir,distance,srcSacNum,delta,\
    28,orignExe=orignExe)
sio.savemat('corrLTest.mat',{'corrLL':corrMat})

fvGD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_norm_g'%i,'file')for i in range(100)}
fvPD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat_norm_p'%i,'file')for i in range(100)}
fvGDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_norm_g'%i,'file')for i in range(100)}
fvPDTest = {'models/ak135%d'%i: d.fv('models/ak135%d_fv_flat_norm_p'%i,'file')for i in range(100)}
disDir = 'disDir/'
if not os.path.exists(disDir):
    os.makedirs(disDir)
i = 0
'''

'''




###
m = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem')
mFlat = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem',isFlat=True)
corrL = [ d.corr().setFromDict(tmpMat) for tmpMat in corrMat]
model = fcn.model()
for i in range(1000):
    corrLTmp = random.sample(corrL,100)
    x,y=d.getTimeDis(corrLTmp,fvGD,T=np.array([5,10,20,30,50,80,100,150,200,250]),sigma=2,maxCount=512)
    model.fit(x,y)

for corr in corrLTmp:
    plt.close()
    #corr = d.corr().setFromDict(tmpMat)
    corr.show(disp,fvPDTest[corr.modelFile])
    plt.savefig('%s%d.jpg'%(disDir,i),dpi=300)
    print(i)
    i+=1
    plt.close()
    if i >20:
        break

outputDir ='predict/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
corrLTest = [ d.corr().setFromDict(tmpMat) for tmpMat in corrMatTest]
corrLTmp = random.sample(corrLTest,10)
time0L=[tmpCorr.timeL[0] for tmpCorr in corrLTmp]
x,y=d.getTimeDis(corrLTmp,fvPDTest,T=np.array([5,10,20,30,50,80,100,150,200,250]),sigma=2,maxCount=512)
model.show(x[:10],y[:10],time0L=time0L,delta=0.5)