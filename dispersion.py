import numpy as np
import scipy
import matplotlib.pyplot as plt
from mathFunc import getDetec,xcorrSimple
from numba import jit,float32, int64
#cps
class layer:
    def __init__(self,vp,vs,rou,z=[0,0],Qp=1200,Qs=600):
        self.z    = np.array(z)
        self.vp   = np.array(vp)
        self.vs   = np.array(vs)
        self.rou  = np.array(rou)
        self.Qp = Qp
        self.Qs = Qs
        self.lamb, self.miu = self.getLame()
        self.zeta = self.getZeta()
        self.xi   = self.getXi()
    def getLame(self):
        miu  = self.vs**2*self.rou
        lamb = self.vp**2*self.rou-2*miu
        return lamb, miu
    def getZeta(self):
        return 1/(self.lamb + 2*self.miu)
    def getXi(self):
        zeta = self.getZeta()
        return 4*self.miu*(self.lamb + self.miu)*zeta
    def getNu(self, k, omega):
        return (k**2-(omega/self.vs.astype(np.complex))**2)**0.5
    def getGamma(self, k,omega):
        return (k**2-(omega/self.vp.astype(np.complex))**2)**0.5
    def getChi(self, k,omega):
        ### is it right
        nu = self.getNu(k, omega)
        return k**2 + nu**2
        #return k**2 + np.abs(nu)**2
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
    def __init__(self, layer0=0, layer1=1,mode='PSV',isTop = False, isBottom = False):
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
    def submat(self,M):
        shape = M.shape
        lenth = shape[0]/2
        newM  = M.reshape([2, lenth, 2, lenth])
        newM  = newM.transpose([0,2,1,3])
        return newM
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
    def setTTRRD(self, surface1 = 0):
        if self.isBottom :
            RRdu1 = np.mat(self.Rdu*0)
            #return 0
        else:
            RRdu1 =  surface1.RRdu
        self.TTd  = (np.mat(np.eye(self.Rud.shape[0])) - np.mat(self.Rud)*np.mat(RRdu1))**(-1)*np.mat(self.Td)
        self.RRdu = np.mat(self.Rdu) + np.mat(self.Tu)*np.mat(RRdu1)*self.TTd
    def setTTRRU(self, surface0 = 0):
        if self.isTop :
            self.RRud = self.Rud
            return 0
        self.TTu  = (np.mat(np.eye(self.Rud.shape[0])) - np.mat(self.Rdu)*np.mat(surface0.RRud))**(-1)*np.mat(self.Tu)
        self.RRud = np.mat(self.Rud) + np.mat(self.Td)*np.mat(surface0.RRud)*self.TTu

class model:
    def __init__(self,modelFile, mode='PSV',getMode = 'norm',layerMode ='norm',layerN=10000):
        #z0 z1 rou vp vs Qkappa Qmu
        #0  1  2   3  4  5      6
        self.modelFile = modelFile
        self.getMode = getMode
        data = np.loadtxt(modelFile)
        layerN=min(data.shape[0],layerN)
        layerL=[None for i in range(layerN)]
        if layerMode == 'norm':
            for i in range(layerN):
                layerL[i] = layer(data[i,3], data[i,4], data[i,2], data[i,:2])
        elif layerMode == 'prem':
            for i in range(layerN):
                #100.0        7.95      4.45      3.38      200.0      80.0
                #0            1         2         3         4          5
                #vp,vs,rou,z=[0,0],Qp=1200,Qs=600
                layerL[i] = layer(data[i,1], data[i,2], data[i,3], np.array([data[i,0],data[(i+1)%layerN,0]]),data[i,4],data[i,5])
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
    def plot(self, omega, dv=0.01):
        #k = np.arange(0,1,dk)
        v, k ,det = self.calList(omega, dv)
        plt.plot(v,np.real(det),'-k')
        plt.plot(v,np.imag(det),'-.k')
        plt.plot(v,np.abs(det),'r')
        plt.show()
    def calList(self,omega,dv=0.01):
        vs0 = self.layerL[1].vs
        vp0 = self.layerL[1].vp
        v = np.arange(vs0-0.499,vs0+2,dv)
        k = omega/v
        det = k.astype(np.complex)*0
        for i in range(k.shape[0]):
            det[i] = self.get(k[i], omega)
        return v, k, det
    def calV(self, omega,order = 0, dv=0.002, DV = 0.008):
        v, k ,det = self.calList(omega, dv)
        iL, detL = getDetec(-np.abs(det), minValue=-0.1, minDelta=int(DV /dv))
        i0 = iL[order]
        v0 = v[i0]
        det0 = -detL[0]
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
    def calDispersion(self, order=0):
        T = np.arange(1,100,5).astype(np.float)
        f = 1/T
        omega = 2*np.pi*f
        v = omega*0
        for i in range(omega.size):
            
            v[i]=np.abs(self.calV(omega[i],order=order))[0]
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
                rou = layer.rou
                if fkMode == 0:
                    vp/=vs
                print('%.2f %.2f %.2f %.2f 1200 600'%(thickness, vs, vp, rou))
                f.write('%.2f %.2f %.2f %.2f 1200 600'%(thickness, vs, vp, rou))
                if i!= self.layerN-1:
                    f.write('\n')

class disp:
    def __init__(self,nperseg=300,noverlap=298,fs=1,halfDt=150,xcorr = xcorrSimple):
        self.nperseg=nperseg
        self.noverlap=noverlap
        self.fs = fs
        self.halfDt = halfDt
        self.halfN = np.int(halfDt*self.fs)
        self.xcorrFunc = xcorrSimple
    def cut(self,data):
        maxI = np.argmax(data)
        i0 = max(maxI - self.halfN,0)
        i1 = min(maxI + self.halfN,data.shape[0])
        return data[i0:i1],i0,i1
    def xcorr(self,data0, data1,isCut=True):
        if isCut:
            data1,i0,i1 = self.cut(data1)
        #print(data0.shape,data1.shape1)
        xx = self.xcorrFunc(data0,data1)
        return xx
    def stft(self,data):
        F,t,zxx = scipy.signal.stft(data,fs=self.fs,nperseg=self.nperseg,\
            noverlap=self.noverlap)
        return F,t,zxx
    def show(self,F,t,zxx,data):
        plt.subplot(2,1,1);plt.pcolor(t,F,np.abs(zxx));plt.subplot(2,1,2);plt.plot(data);plt.show()
    def test(self,data0,data1,isCut=True):
        xx = self.xcorr(data0,data1,isCut=isCut)
        F,t,zxx = self.stft(xx)
        self.show(F,t,zxx,xx)


