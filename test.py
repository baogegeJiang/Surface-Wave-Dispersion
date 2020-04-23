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
#是否需要考虑展平变化的影响
m = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem')
m.covert2Fk(0)
m.covert2Fk(1)
F,v =m.calDispersion(order=0,calMode='fast',threshold=0.1,T=3*10**np.arange(0,2.1,0.2))
plt.axes(xscale = "log");plt.plot(1/F,v);plt.show()
orignExe='/home/jiangyr/program/fk/'
f = fk.FK(orignExe=orignExe)
sacsL,sacNamesL = f.test(distance=[1000,1100],modelFile='prem',fok='/f',dt=1,depth=20,\
    expnt=9,dura=20,dk=0.1,srcSac='../sourceSac.sac')
disp = d.disp(nperseg=100,noverlap=98,fs=1,halfDt=150,xcorrFunc = mathFunc.xcorrSimple)
corr=disp.testSac(sacsL[0][1][0],sacsL[0][0][0],fTheor=F,vTheor=v);plt.show()
#d.genModel(modelFile = 'prem',N=1000,perD = 0.10,depthMul=2)
absPath = '/home/jiangyr/home/Surface-Wave-Dispersion/'

def calFv(iL):
    for i in iL:
        modelFile = 'models/prem%d'%i
        print(i)
        d.genFvFile(modelFile,fvFile='',mode='PSV',getMode = 'norm',layerMode ='prem',layerN=20,calMode='fast',\
            T=np.array([5,10,20,30,50,80,100,150,200,250,300]),isFlat=True)


srcSacNum = 100
delta=0.5
srcSacDir='/home/jiangyr/Surface-Wave-Dispersion/srcSac/'
f = fk.FK(orignExe=orignExe)
fk.genSourceSacs(f,srcSacNum,delta,srcSacDir = srcSacDir,time=50)
disp = d.disp(nperseg=200,noverlap=196,fs=1/delta,halfDt=150,xcorrFunc = mathFunc.xcorrSimple)
distance=np.array([800,900,1000,1100,1200,1300,1400,1500])
def singleFk(f,iL,corrLL,index):
    for i in iL:
        modelFile = 'models/prem%d'%i
        m = d.model(modelFile=modelFile,mode='PSV',layerN=20,layerMode ='prem',isFlat=True)
        m.covert2Fk(0)
        m.covert2Fk(1)
        dura = np.random.rand()*10+20
        depth= int(np.random.rand()*20+10)
        M=np.array([3e25,0,0,0,0,0,0])
        M[1:] = np.random.rand(6)
        srcSacIndex = int(np.random.rand()*srcSacNum*0.999)
        rise = 0.1+0.3*np.random.rand()
        sacsL, sacNamesL= f.test(distance=distance+np.round((np.random.rand(distance.size)-0.5)*80),\
            modelFile=modelFile,fok='/k',dt=delta,depth=depth,expnt=10,dura=dura,dk=0.1,\
            azimuth=[0,int(6*(np.random.rand()-0.5))],M=M,rise=rise,srcSac=fk.getSourceSacName(srcSacIndex,delta,\
                srcSacDir = srcSacDir))
        corrLL[index] += d.corrSacsL(disp,sacsL,sacNamesL,modelFile=modelFile,\
            srcSac=fk.getSourceSacName(srcSacIndex,delta,srcSacDir = srcSacDir))
f = fk.FK(orignExe=orignExe)
pL = []
pN = 20
for i in range(pN):
    pL.append(  multiprocessing.Process(target=calFv, args=(range(i,100,pN), )))
    pL[-1].start()

for p in pL:
    p.join()

corrMat = fk.multFK(2,singleFk,2,orignExe=orignExe)
sio.savemat('corrL.mat',{'corrLL':corrMat})

fvD = {'models/prem%d'%i: d.fv('models/prem%d_fv_flat'%i,'file')for i in range(100)}

disDir = 'disDir/'
if not os.path.exists(disDir):
    os.makedirs(disDir)
i = 0
for tmpMat in corrMat:
    plt.close()
    corr = d.corr().setFromDict(tmpMat)
    corr.show(disp,fvD[corr.modelFile])
    plt.savefig('%s%d.jpg'%(disDir,i),dpi=300)
    print(i)
    i+=1
    plt.close()
    if i >10:
        break




###
m = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem')
mFlat = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem',isFlat=True)

