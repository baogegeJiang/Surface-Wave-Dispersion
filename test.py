import dispersion as d
import fk
from imp import reload
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
m = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem')
m.covert2Fk(0)
m.covert2Fk(1)
F,v =m.calDispersion(order=0,calMode='fast',threshold=0.1,T=3*10**np.arange(0,2.1,0.2))
plt.axes(xscale = "log");plt.plot(1/F,v);plt.show()
orignExe='/home/jiangyr/program/fk/'
f = fk.FK(orignExe=orignExe)
sacsL = f.test(distance=[1000,1100],modelFile='prem',fok='/f',dt=1,depth=20,expnt=9,dura=20,dk=0.1)
disp = d.disp(nperseg=100,noverlap=98,fs=1,halfDt=150)
disp.testSac(sacsL[1][0],sacsL[0][0],fTheor=F,vTheor=v);plt.show()
#d.genModel(perD=0.05, depthMul=4,N=1000)
for i in range(1000):
    modelFile = 'models/prem%d'%i
    d.genFvFile(modelFile,fvFile='',mode='PSV',getMode = 'norm',layerMode ='prem',layerN=20,calMode='fast',\
    T=np.array([5,10,20,30,50,80,100,150,200,250,300]))

def calFv(iL):
    for i in iL:
        modelFile = 'models/prem%d'%i
        print(i)
        d.genFvFile(modelFile,fvFile='',mode='PSV',getMode = 'norm',layerMode ='prem',layerN=20,calMode='fast',\
            T=np.array([5,10,20,30,50,80,100,150,200,250,300]))

pL = []
pN = 20
for i in range(pN):
    pL.append(  multiprocessing.Process(target=calFv, args=(range(i,1000,pN), )))
    pL[-1].start()


i=0
for p in pL:
    p.join()
    print('######',i)
    i+=1

corrL = []
f = fk.FK(orignExe=orignExe)
for i in range(1):
    modelFile = 'models/prem%d'%i
    m = d.model(modelFile=modelFile,mode='PSV',layerN=20,layerMode ='prem')
    m.covert2Fk(0)
    m.covert2Fk(1)
    dura = np.random.rand()*10+10
    depth= int(np.random.rand()*40+10)
    M=np.array([3e20,0,0,0,0,0,0])
    M[1:] = np.random.rand(6)
    sacsL = f.test(distance=[500,600,700,800,900,1000,1100,1200,1300,1400,1500],\
        modelFile=modelFile,fok='/f',dt=1,depth=depth,expnt=9,dura=dura,dk=0.1,\
        azimuth=[0,2,4,6],M=M)
    corrL += d.corrSacsL(disp,sacsL)
