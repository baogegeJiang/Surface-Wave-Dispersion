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
d.genModel()
for i in range(1000):
    modelFile = 'models/prem%d'%i
    d.genFvFile(modelFile,fvFile='',mode='PSV',getMode = 'norm',layerMode ='prem',layerN=20,calMode='fast',\
    T=np.array([5,10,20,30,50,80,100,150,200,250,300]))

def calFv(iL):
    for i in iL:
        modelFile = 'models/prem%d'%i
        d.genFvFile(modelFile,fvFile='',mode='PSV',getMode = 'norm',layerMode ='prem',layerN=20,calMode='fast',\
            T=np.array([5,10,20,30,50,80,100,150,200,250,300]))
pL=[]
for i in range(10):
    pL.append(  multiprocessing.Process(target=calFv, args=(range(i,100,10), )))
    pL[-1].start()
i=0
for p in pL:
    p.join()
    print(i)
    i+=1