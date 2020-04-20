import numpy as np
import scipy
import matplotlib.pyplot as plt
from mathFunc import getDetec,xcorrSimple,xcorrComplex
from numba import jit,float32, int64
from scipy import fftpack,interpolate
from fk import FK
import os 
'''
the specific meaning of them you can find in Chen Xiaofei's paper
(A systematic and efficient method of computing normal modes for multilayered half-space)
'''
class layer:
    '''
    class for layered media;
    the p velocity(vp), s velocity(vs), density(rho), [top depth, bottom depth](z) is needed; 
    p and s 's Q(Qp, Qs) is optional(default is 1200,600)
    After specifying the above parameters, the lame parameter(lambda, miu), zeta and xi would
     be calculate as class's attributes. 
    '''
    def __init__(self,vp,vs,rho,z=[0,0],Qp=1200,Qs=600):
        self.z    = np.array(z)
        self.vp   = np.array(vp)
        self.vs   = np.array(vs)
        self.rho  = np.array(rho)
        self.Qp = Qp
        self.Qs = Qs
        self.lamb, self.miu = self.getLame()
        self.zeta = self.getZeta()
        self.xi   = self.getXi()
    @jit
    def getLame(self):
        miu  = self.vs**2*self.rho
        lamb = self.vp**2*self.rho-2*miu
        return lamb, miu
    @jit
    def getZeta(self):
        return 1/(self.lamb + 2*self.miu)
    @jit
    def getXi(self):
        zeta = self.getZeta()
        return 4*self.miu*(self.lamb + self.miu)*zeta
    @jit
    def getNu(self, k, omega):
        return (k**2-(omega/self.vs.astype(np.complex))**2)**0.5
    @jit
    def getGamma(self, k,omega):
        return (k**2-(omega/self.vp.astype(np.complex))**2)**0.5
    @jit
    def getChi(self, k,omega):
        ### is it right
        nu = self.getNu(k, omega)
        return k**2 + nu**2
        #return k**2 + np.abs(nu)**2
    @jit
    def getEA(self, k,omega, z,mode = 'PSV'):
        nu    = self.getNu(k, omega)
        gamma = self.getGamma(k, omega)
        chi   = self.getChi(k,omega)
        alpha = self.vp
        beta  = self.vs
        miu   = self.miu
        if mode == 'PSV':
            E = 1/omega*np.array(\
                  [[ alpha*k,              beta*nu,          alpha*k,             beta*nu],\
                  [  alpha*gamma,          beta*k,           -alpha*gamma,        -beta*k],  \
                  [  -2*alpha*miu*k*gamma, -beta*miu*chi,    2*alpha*miu*k*gamma, beta*miu*chi],
                  [  -alpha*miu*chi,       -2*beta*miu*k*nu, -alpha*miu*chi,      -2*beta*miu*k*nu]])
            A = np.array(\
                [[np.exp(-gamma*(z-self.z[0])), 0,                         0,                           0],\
                 [0,                            np.exp(-nu*(z-self.z[0])), 0,                           0],\
                 [0,                            0,                         np.exp(gamma*(z-self.z[1])), 0],\
                 [0,                            0,                         0,                           np.exp(nu*(z-self.z[1]))]\
                ])
        elif mode == 'SH':
            E = np.array(\
                [[1,       1],\
                [ -miu*nu, miu*nu]])
            A = np.array(\
                [[np.exp(-nu*(z-self.z[0])), 0],\
                [ 0,                         np.exp(nu*(z-self.z[1]))]])
        return E, A

class surface:
    '''
    class for surface of layer
    the layers above and beneath(layer0, layer1) is needed
    Specify the bool parameters isTop and isBottom, if the surface is the first or last one； the default is false
    the default waveform mode(mode) is 'PSV', you can set it to 'SH'
    '''
    def __init__(self, layer0, layer1,mode='PSV',isTop = False, isBottom = False):
        self.layer0 = layer0
        self.layer1 = layer1
        #if not isTop:
        self.z      = layer0.z[-1]
        self.Td       = 0
        self.Tu       = 0
        self.Rud      = 0
        self.Rdu      = 0
        self.TTd      = 0
        self.TTu      = 0
        self.RRud     = 0
        self.RRdu     = 0
        self.mode     = mode
        self.isTop    = isTop
        self.isBottom = isBottom
        self.E = [None,None]
        self.A = [None,None]
    @jit
    def submat(self,M):
        shape = M.shape
        lenth = int(shape[0]/2)
        newM  = M.reshape([2, lenth, 2, lenth])
        newM  = newM.transpose([0,2,1,3])
        return newM
    @jit
    def setTR(self, k, omega):
        E0, A0 = self.layer0.getEA(k, omega, self.z, self.mode)
        E1, A1 = self.layer1.getEA(k, omega, self.z,self.mode)
        E0     = self.submat(E0)
        #print(E0[0][0].shape)
        E1     = self.submat(E1)
        A0     = self.submat(A0)
        A1     = self.submat(A1)
        self.E = [E0, E1]
        self.A = [A0, A1]
        EE0    = self.toMat([[E1[0][0],   -E0[0][1]],\
            [                 E1[1][0],   -E0[1][1]]])
        EE1    = self.toMat([[E0[0][0],   -E1[0][1]],\
            [                 E0[1][0],   -E1[1][1]]])
        AA     = self.toMat([[A0[0][0],   A0[0][0]*0],\
            [                 A1[0][0]*0, A1[1][1]]])
        #print(AA)
        TR     = EE0**(-1)*EE1*AA
        TR     = self.submat(np.array(TR))
        self.Td  = TR[0][0]
        self.Rdu = TR[1][0]
        self.Rud = TR[0][1]
        self.Tu  = TR[1][1]
        if self.isTop:
            self.Rud = -E1[1][0]**(-1)*E1[1][1]*(A1[1][1])
            self.Td = self.Rud*0
            self.Tu = self.Rud*0
    @jit
    def toMat(self,l):
        shape0 = len(l)
        shape1 = len(l[0])
        shape = np.zeros(2).astype(np.int64)
        #print(l[0][0].shape)
        shape[0]  = l[0][0].shape[0]
        shape[1]  = l[0][0].shape[1]
        SHAPE  = shape+0
        SHAPE[0] *=shape0
        SHAPE[1] *=shape1
        M = np.zeros(SHAPE,np.complex)
        for i in range(shape0):
            for j in range(shape1):
                i0 = i*shape[0]
                i1 = (i+1)*shape[0]
                j0 = j*shape[1]
                j1 = (j+1)*shape[1]
                #print(i0,i1,j0,j1)
                M[i0:i1,j0:j1] = l[i][j]
        return np.mat(M)
    @jit
    def setTTRRD(self, surface1 = 0):
        if self.isBottom :
            RRdu1 = np.mat(self.Rdu*0)
            #return 0
        else:
            RRdu1 =  surface1.RRdu
        self.TTd  = (np.mat(np.eye(self.Rud.shape[0])) - np.mat(self.Rud)*np.mat(RRdu1))**(-1)*np.mat(self.Td)
        self.RRdu = np.mat(self.Rdu) + np.mat(self.Tu)*np.mat(RRdu1)*self.TTd
    @jit
    def setTTRRU(self, surface0 = 0):
        if self.isTop :
            self.RRud = self.Rud
            return 0
        self.TTu  = (np.mat(np.eye(self.Rud.shape[0])) - np.mat(self.Rdu)*np.mat(surface0.RRud))**(-1)*np.mat(self.Tu)
        self.RRud = np.mat(self.Rud) + np.mat(self.Td)*np.mat(surface0.RRud)*self.TTu

class model:
    '''
    class for layered media model
    modeFile is the media parameter model File, there are tow mods
    if layerMode == 'norm':
       '0    18  2.80  6.0 3.5'
       layer's top depth, layer's bottom depth, density, p velocity, svelocity
    if layerMode =='prem':
        '0.00       5.800     3.350       2.800    1400.0     600.0'
        depth,  p velocity, s velocity, density,    Qp,        Qs
    mode is for PSV and SH
    getMode is the way to get phase velocity:
        norm is enough to get phase velocity
        new is to get fundamental phase velocity for PSV
    '''
    def __init__(self,modelFile, mode='PSV',getMode = 'norm',layerMode ='norm',layerN=10000):
        #z0 z1 rho vp vs Qkappa Qmu
        #0  1  2   3  4  5      6
        self.modelFile = modelFile
        self.getMode = getMode
        data = np.loadtxt(modelFile)
        layerN=min(data.shape[0],layerN+1)
        layerL=[None for i in range(layerN)]
        if layerMode == 'norm':
            layerL[0] = layer(1.7, 1, 0.0001,[-100,0])
            for i in range(1,layerN):
                layerL[i] = layer(data[i-1,3], data[i-1,4], data[i-1,2], data[i-1,:2])
        elif layerMode == 'prem':
            layerL[0] = layer(1.7, 1, 0.0001,[-100,0])
            for i in range(1,layerN):
                #100.0        7.95      4.45      3.38      200.0      80.0
                #0            1         2         3         4          5
                #vp,vs,rho,z=[0,0],Qp=1200,Qs=600
                layerL[i] = layer(data[i-1,1], data[i-1,2], data[i-1,3], np.array([data[i-1,0],data[(i+1-1)%layerN,0]]),data[i-1,4],data[i-1,5])
        surfaceL = [None for i in range(layerN-1)]
        for i in range(layerN-1):
            isTop = False
            isBottom = False
            if i == 0:
                isTop = True
            if i == layerN-2:
                isBottom = True
            surfaceL[i] = surface(layerL[i], layerL[i+1], mode, isTop, isBottom)
        self.layerL = layerL
        self.surfaceL = surfaceL
        self.layerN = layerN
    @jit
    def set(self, k,omega):
        for s in self.surfaceL:
            s.setTR(k,omega)
        for i in range(self.layerN-1-1,-1,-1):
            #print(i)
            s = self.surfaceL[i]
            if i == self.layerN-1-1:
                s.setTTRRD(self.surfaceL[0])
            else:
                s.setTTRRD(self.surfaceL[i+1])
        for i in range(self.layerN-1):
            #print(i)
            s = self.surfaceL[i]
            if i == 0:
                s.setTTRRU(self.surfaceL[0])
            else:
                s.setTTRRU(self.surfaceL[i-1])
    @jit
    def get(self, k, omega):
        self.set(k, omega)
        RRud0 = self.surfaceL[0].RRud
        RRdu1 = self.surfaceL[1].RRdu
        if self.getMode == 'norm':
            M = np.mat(np.eye(RRud0.shape[0])) - RRud0*RRdu1
        elif self.getMode == 'new':
            #-E1[1][0]**(-1)*E1[1][1]*(A1[1][1])
            M = self.surfaceL[0].E[1][1][0]+self.surfaceL[0].E[1][1][1]*self.surfaceL[0].A[1][1][1]*RRdu1
        return np.linalg.det(M)
    @jit
    def plot(self, omega, dv=0.01):
        #k = np.arange(0,1,dk)
        v, k ,det = self.calList(omega, dv)
        plt.plot(v,np.real(det),'-k')
        plt.plot(v,np.imag(det),'-.k')
        plt.plot(v,np.abs(det),'r')
        plt.show()
    @jit
    def calList(self,omega,dv=0.01):
        vs0 = self.layerL[1].vs
        vp0 = self.layerL[1].vp
        v = np.arange(vs0-0.499,vs0+5,dv)
        k = omega/v
        det = k.astype(np.complex)*0
        for i in range(k.shape[0]):
            det[i] = self.get(k[i], omega)
        return v, k, det
    @jit
    def calV(self, omega,order = 0, dv=0.002, DV = 0.008,calMode='norm',threshold=0.1):
        if calMode =='norm':
            v, k ,det = self.calList(omega, dv)
            iL, detL = getDetec(-np.abs(det), minValue=-0.1, minDelta=int(DV /dv))
            i0 = iL[order]
            v0 = v[i0]
            det0 = -detL[0]
        elif calMode == 'fast':
             v0,det0=self.calVFast(omega,order=order,dv=dv,DV=DV,threshold=threshold)
        '''
        ddv = 0.001  
        for i in range(5):
            step = 1e-3*(5-i)
            v1 = v0 + ddv
            det1 = np.abs(self.get(omega/v1, omega))
            k = (det1-det0)/ddv
            v0 = v0 - k*step
            print(k)
            det0 = np.abs(self.get(omega/v1, omega))
        '''
        return v0,det0
    @jit
    def calVFast(self,omega,order=0,dv=0.01,DV=0.008,threshold=0.1):
        v = self.layerL[1].vs+1e-8
        v0 = v
        det0=10
        for i in range(10000):
            v1 = i*dv+v
            det1 =np.abs(self.get(omega/v1, omega))
            if  det1<threshold and det1 < det0:
                v0 = v1
                det0 = det1
            if det0 <threshold and det1>det0:
                return v0, det0
    @jit
    def calDispersion(self, order=0,calMode='norm',threshold=0.1,T= np.arange(1,100,5).astype(np.float)):
        f = 1/T
        omega = 2*np.pi*f
        v = omega*0
        for i in range(omega.size):
            v[i]=np.abs(self.calV(omega[i],order=order,calMode=calMode,threshold=threshold))[0]
            print(omega[i],v[i])
        return f,v
    def test(self):
        self.plot(2*np.pi)
    def testDispersion(self):
        f,v = self.calDispersion()
        plt.plot(f,v)
        plt.show()
    def compare(self,dv=0.01):
        self.getMode = 'norm'
        v, k ,det = self.calList(6.28, dv)
        plt.plot(v,np.abs(det)/np.abs(det).max(),'k')
        self.getMode = 'new'
        v, k ,det = self.calList(6.28, dv)
        plt.plot(v,np.abs(det)/np.abs(det).max(),'r')
        plt.show()
    def covert2Fk(self, fkMode=0):
        if fkMode == 0:
            filename = self.modelFile+'fk0'
        else:
            filename = self.modelFile+'fk1'
        with open(filename,'w+') as f:
            for i in range(1,self.layerN):
                layer = self.layerL[i]
                thickness = layer.z[1] - layer.z[0]
                vp = layer.vp.copy()
                vs = layer.vs.copy()
                rho = layer.rho
                if fkMode == 0:
                    vp/=vs
                print('%.2f %.2f %.2f %.2f 1200 600'%(thickness, vs, vp, rho))
                f.write('%.2f %.2f %.2f %.2f 1200 600'%(thickness, vs, vp, rho))
                if i!= self.layerN-1:
                    f.write('\n')



class disp:
    '''
    traditional method to calculate the dispersion curve
    then should add some sac to handle time difference
    '''
    def __init__(self,nperseg=300,noverlap=298,fs=1,halfDt=150,xcorr = xcorrComplex):
        self.nperseg=nperseg
        self.noverlap=noverlap
        self.fs = fs
        self.halfDt = halfDt
        self.halfN = np.int(halfDt*self.fs)
        self.xcorrFunc = xcorrSimple
    @jit
    def cut(self,data):
        maxI = np.argmax(data)
        i0 = max(maxI - self.halfN,0)
        i1 = min(maxI + self.halfN,data.shape[0])
        print(i0,i1)
        return data[i0:i1],i0,i1
    @jit
    def xcorr(self,data0, data1,isCut=True):
        if isCut:
            data1,i0,i1 = self.cut(data1)
        #print(data0.shape,data1.shape1)
        xx = self.xcorrFunc(data0,data1)
        return xx,i0,i1
    @jit
    def stft(self,data):
        F,t,zxx = scipy.signal.stft(data,fs=self.fs,nperseg=self.nperseg,\
            noverlap=self.noverlap)
        zxx /= np.abs(zxx).max(axis=1,keepdims=True)
        return F,t,zxx
    def show(self,F,t,zxx,data,timeL,isShow=True):
        plt.subplot(2,1,1);plt.pcolor(t,F,np.abs(zxx));plt.subplot(2,1,2);plt.plot(timeL,data);
        if isShow:
            plt.show()
    def sacXcorr(self,sac0,sac1,isCut=True):
        fs = sac0.stats['sampling_rate']
        self.fs=fs
        self.halfN = np.int(self.halfDt*self.fs)
        data0 = sac0.data
        time0 = sac0.stats.starttime.timestamp
        dis0  = sac0.stats['sac']['dist']
        data1 = sac1.data
        time1 = sac1.stats.starttime.timestamp
        dis1  = sac1.stats['sac']['dist']
        xx,i0,i1 = self.xcorr(data0,data1,isCut)
        time1New = time1+i0/fs
        dTime =  time0 -time1New
        timeL = np.arange(xx.size)/fs+dTime
        dDis = dis0 - dis1
        return corr(xx,timeL,dDis,fs)
    def test(self,data0,data1,isCut=True):
        xx = self.xcorr(data0,data1,isCut=isCut)
        F,t,zxx = self.stft(xx)
        self.show(F,t,zxx,xx)
    def testSac(self,sac0,sac1,isCut=True,fTheor=[],vTheor=[]):
        xx,timeL,dDis,fs = self.sacXcorr(sac0,sac1,isCut=True).output()
        F,t,zxx = self.stft(xx)
        print(t)
        t = t+timeL[0]+0*self.nperseg/self.fs
        self.show(F,t,zxx,xx,timeL,isShow=False)
        if len(fTheor)>0:
            timeTheorL =dDis/vTheor
            plt.subplot(2,1,1);plt.plot(timeTheorL,fTheor)
        return xx, zxx, F, t

class fv:
    '''
    class for dispersion result
    it have two attributes f and v, each element in v accosiate with an 
     element in v 
    '''
    def __init__(self,input,mode='num'):
        if mode == 'num':
            self.f = input[0]
            self.v = input[1]
        if mode == 'file':
            fvM = np.loadtxt(input)
            self.f = fvM[:,0]
            self.v = fvM[:,1]
        self.interp = self.genInterp()
    def genInterp(self):
        return interpolate.interp1d(self.f,self.v,kind='linear')
    def __call__(self,f):
        return self.interp(f)
    def save(self,filename):
        np.savetxt(filename, np.concatenate([self.f.reshape([-1,1]),\
            self.v.reshape([-1,1])],axis=1))


class corr:
    """docstring for """
    def __init__(self,xx=0,timeL=0,dDis=0,fs=0,az=np.array([0,0]),dura=0,M=np.array([0,0,0,0,0,0,0])\
        ,dis=np.array([0,0]),dep = 10,modelFile=''):
        self.xx    = xx 
        self.timeL = timeL
        self.dDis  = dDis
        self.fs    = fs
        self.az    = az
        self.dura  = dura
        self.M     = M
        self.dis   = dis
        self.dep   = dep
        self.modelFile=modelFile
    def output(self):
        return self.xx,self.timeL,self.dDis,self.fs
    def toDict(self):
        return {'xx':self.xx, 'timeL':self.timeL, 'dDis':self.dDis, 'fs':self.fs,\
        'az':self.az, 'dura':self.dura,'M':self.M,'dis':self.dis,'dep':self.dep,\
        'modelFile':self.modelFile}
    def setFromFile(self,file):
        mat        = scipy.io.load(file)
        self.xx    = mat['xx'] 
        self.timeL = mat['timeL']
        self.dDis  = mat['dDis']
        self.fs    = mat['fs']
        self.az    = mat['az']
        self.dura  = mat['dura']
        self.M     = mat['M']
        self.dis   = mat['dis']
        self.dep   = mat['dep']
        self.modelFile = mat['modelFile']
    def save(self,fileName):
        scipy.io.savemat(fileName,self.toDict())

        
def genModel(modelFile = 'prem',N=100,perD = 0.10,depthMul=2):
    modelDir = 'models/'
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)
    #800.0       11.0       6.13      4.46      740.0     312.0
    model0 = np.loadtxt(modelFile)
    for i in range(N):
        model = model0.copy()
        depthLast = -10
        for j in range(model.shape[0]):
            depth0 = model[j,0]
            depth = max(depthLast,depth0 + (depth0-depthLast)*perD*depthMul*(2*np.random.rand()-1))
            depth0= depth
            model[j,0]=depth
            for k in range(1,6):
                model[j,k]*=1+perD*(2*np.random.rand()-1)
        np.savetxt('%s/%s%d'%(modelDir,modelFile,i),model)

def genFvFile(modelFile,fvFile='',mode='PSV',getMode = 'norm',layerMode ='prem',layerN=20,calMode='fast',\
    T=np.array([0.5,1,5,10,20,30,50,80,100,150,200,250,300])):
    if len(fvFile) ==0:
        fvFile='%s_fv'%modelFile
    m = model(modelFile,mode=mode,getMode=getMode,layerMode=layerMode,layerN=layerN)
    f,v=m.calDispersion(order=0,calMode=calMode,threshold=0.1,T=T)
    f = fv([f,v],'num')
    f.save(fvFile)
