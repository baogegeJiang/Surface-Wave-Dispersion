import dispersion as d
import fk
from imp import reload
import matplotlib.pyplot as plt
import numpy as np
m = d.model(modelFile='prem',mode='PSV',layerN=20,layerMode ='prem')
m.covert2Fk(0)
m.covert2Fk(1)
F,v =m.calDispersion(order=0,calMode='fast',threshold=0.1,T=3*10**np.arange(0,2.1,0.2))
plt.axes(xscale = "log");plt.plot(1/F,v);plt.show()
f = fk.FK()
sacsL = f.test(distance=[1000,1100],modelFile='premfk',fok='/f',dt=1,depth=20,expnt=9,dura=20,dk=0.1)
disp = d.disp(nperseg=100,noverlap=98,fs=1,halfDt=150)
disp.testSac(sacsL[1][0],sacsL[0][0],fTheor=F,vTheor=v);plt.show()
d.genModel()