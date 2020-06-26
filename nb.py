import numpy as np
import struct
import sys
from scipy import signal
import os
import obspy
from matplotlib import pyplot as plt
from scipy import ndimage
if sys.byteorder =='little':
    h = '<'
else:
    h = '>'
class Model:
    #right
    def __init__(self):
        pass
    def to2D(self,x,z):
        nz = z.size
        nx = x.size
        one = np.ones([nz,nx])
        vp = one*6.0
        vs = vp/1.7
        den= one*3.0
        return vp,vs,den

class Model2:
    def __init__(self):
        pass
    def to2D(self,x,z):
        nz = z.size
        nx = x.size
        one = np.ones([nz,nx])
        vp = one*6.0
        vp[int(nz/2):]/=0.9
        vs = vp/1.7
        den= one*3.0
        return vp,vs,den

class Model3:
    def __init__(self):
        pass
    def to2D(self,x,z):
        nz = z.size
        nx = x.size
        one = np.ones([nx,nz])
        vp = one*6.0
        vp[:,int(nz/2):]/=2
        vs = vp/1.7
        den= one*3.0
        return vp,vs,den

class Model4:
    def __init__(self):
        pass
    def to2D(self,x,z):
        nz = z.size
        nx = x.size
        one = np.ones([nz,nx])
        vp = one*6.0
        vp[int(nz/3):]/=0.9
        vp[2*int(nz/3):]/=0.9
        vs = vp/1.6
        den= one*3.0
        return vp,vs,den

def read(file):
    with open(file, 'rb') as f:
        s = f.read()
    nT = int(toNum('i',8,s)[7])
    perSize = (nT+21)*struct.calcsize(h+'f')
    nRec = int(len(s)/perSize)
    data = toNum('f',nRec*(nT+21),s)
    data = data.reshape([nRec,nT+21])
    b  = data[:,21:]
    xs = data[:,0]
    zs = data[:,2]
    xr = data[:,3]
    zr = data[:,5]
    dt = data[:,8]
    gain = data[:,12]
    return Records(b,xs,zs,xr,zr,dt,gain)

def toNum(TYPE,n,s):
    typeL = h+TYPE*n
    size = struct.calcsize(typeL)
    return np.array(struct.unpack(typeL,s[:size]))

def toB(v,TYPE='f'):
    typeL= h+ TYPE*v.size
    return struct.pack(typeL,*(v.reshape([-1]).tolist()))

class Records:
    def __init__(self,data,xs,zs,xr,zr,dt,gain):
        self.data = data
        self.xs   = xs[0::2]
        self.zs   = zs[0::2]
        self.xr   = xr[0::2]
        self.dt   = dt[0::2]
        self.gain  = gain
        self.nt = self.data.shape[1]
        self.t  = self.dt[0]*(np.arange(self.nt)+(np.pi**2)/16)
    def __call__(self):
        bt = (self.data[1::2]+self.data[0::2])/2
        btNew = bt*0
        for i in  range(bt.shape[0]):
            btNew[i] = signal.convolve(1/(self.t**0.5),bt[i])[:self.nt]
        btNew[:,1:]=btNew[:,1:]-btNew[:,0:-1]

        btp = (self.data[1::2]-self.data[0::2])/2*self.gain[0]
        btpNew = btp*0
        for i in  range(btp.shape[0]):
            btpNew[i] = signal.convolve(1/(self.t**0.5),btp[i])[:self.nt]
        bts=np.sign(btNew)*np.sqrt(abs(btNew*btpNew))
        for i in range(bts.shape[0]):
            sqrtR = np.abs(self.xr[i]-self.xs[i])**0.5
            btNew[i]/=sqrtR
            btpNew[i]/=sqrtR
            bts[i]/=sqrtR
        return btNew,btpNew,bts

class NB:
    def __init__(self,para={},mainDir ='gpu_2d/'):
        self.para0={\
        'nx':16384,
        'nz':2048,
        'model':'example',
        'nt':160000,
        'gpuid':1,
        'xs':500,
        'zs':50,
        'h': 1,
        'dt':0.02,
        'srctype':'1',
        'srctime':'g',
        'alpha':-20,
        'trap1':0,
        'trap2':0,
        'trap3':0,
        'strike':0.1,
        'dip':20.1,
        'rake':1.1,
        'azimuth':180.1,
        'npml':64,
        'npml':64,
        'pml_r':1e-11,
        'pml_dt':0.005,
        'pml_v':30.0,
        'pml_fc':2,
        'itrecord':1  ,    
        'output':'output',
        'nrec':60,
        'ixrec0':600,
        'izrec0_u':1,    
        'izrec0_w':1,
        'izrec0_v':1,
        'idxrec':150 ,
        'idzrec':0  ,
        'ntsnap':1000000,
        'usetable':0,
        'itprint' :1000
        }
        self.update(para)
        self.mainDir = mainDir
        if not  os.path.exists(self.mainDir):
            os.makedirs(self.mainDir)
        if not os.path.exists(self.runDir()):
            os.makedirs(self.runDir())
        if not os.path.exists(self.resDir()):
            os.makedirs(self.resDir())
    def update(self,para):
        for key in para:
            self.para0[key] = type(self.para0[key])(para[key])
    def runDir(self):
        return self.mainDir+'run/'
    def resDir(self):
        return self.mainDir+'res/'
    def writePar(self,parName=''):
        if parName=='':
            parName = self.runDir()+self.para0['model']+'.par'
        with open(parName,'w') as f:
            for key in self.para0:
                f.write(key+'='+str(self.para0[key])+'\n')
    def writeModel(self,vp,vs,den):
        with open(self.runDir()+self.para0['model']+'.vp','wb') as f:
            f.write(toB(vp))
        with open(self.runDir()+self.para0['model']+'.vs','wb') as f:
            f.write(toB(vs))
        with open(self.runDir()+self.para0['model']+'.den','wb') as f:
            f.write(toB(den))
    def test(self,model=Model(),para={},isFilter=False,isPlot=False):
        self.update(para)
        para = self.para0
        nx = para['nx']
        nz = para['nz']
        x = np.arange(nx)*para['h']
        z = np.arange(nz)*para['h']
        vp,vs,den=model.to2D(x,z)
        if isFilter:
            vp = ndimage.gaussian_filter(vp,10,mode='nearest')
            vs = ndimage.gaussian_filter(vs,10,mode='nearest')
            den = ndimage.gaussian_filter(den,10,mode='nearest')
        with open('models/'+para['output']+'_smooth','w') as f:
            for i in range(nz):
                f.write('%f %f %f %f 1200 600\n'%(z[i],vp[i,0],vs[i,0],den[i,0]*1e3))
        #return
        if isPlot:
            self.plot(x,z,vp,vs,den)
        self.writeModel(vp,vs,den)
        self.writePar()
        self.run()
        self.loadRes()
        self.saveRes()

    def plot(self,x,z,vp,vs,den):
        plt.close()
        mL=[vp,vs,den]
        for i in range(3):
            m=mL[i]
            plt.subplot(3,1,i+1)
            plt.pcolor(x,z,m)
            plt.ylabel('z/km')
            plt.xlabel('x/km')
            plt.xlim([x.min(),x.max()])
            plt.ylim([z.max(),z.min()])
        plt.savefig(self.resDir()+self.para0['output']+'/''model.jpg',dpi=300)


    def run(self):
        for exe in ['nbpsv2d','nbsh2d']:
            cmd = 'cd %s; %s par=%s.par'%(self.runDir(),exe,self.para0['model'])
            os.system(cmd)
    def loadRes(self,compL='VWU'):
        self.recordsL = [read(self.runDir()+self.para0['output']+'_%s.isis'%comp) for comp in compL]
    def saveRes(self):
        recordsL = self.recordsL
        para0 = self.para0
        compL = 'ENZ'
        d2k = 111.19
        t = obspy.UTCDateTime(0)
        quakeDir = self.resDir()+para0['output']+'/'
        if not os.path.exists(quakeDir):
            os.makedirs(quakeDir)
        for compIndex in range(3):
            records = recordsL[compIndex]
            bt,btp,bts = records()
            comp = compL[compIndex]
            for i in range(records.xs.shape[0]):
                filename=quakeDir+'%d.BH%s.SAC'%(i,comp)
                km = np.abs(records.xs[i]-records.xr[i])
                deg = km/d2k
                az = para0['azimuth']
                header = {'kstnm': 'NB%d'%i, 'kcmpnm': 'BH%s'%comp, 'stla': deg,\
                'stlo': 0,'evla': 0, 'evlo': 0, 'evdp': records.zs[i],\
                'nzyear': t.year,'nzjday': t.julday, 'nzhour': t.hour, \
                'nzmin': t.minute,'nzsec': t.second,'nzmsec': int(t.microsecond/1e3)\
                , 'delta': records.dt[i],'b':0,'dist':km}
                sac = obspy.io.sac.sactrace.SACTrace(data=bt[i],**header).to_obspy_trace()
                sac.write(filename)
        nameStr = ''
        
        for i in range(recordsL[0].xs.shape[0]):
            for comp in compL:
                nameStr+='%s/%d.BH%s.SAC '%(quakeDir,i,comp)
            nameStr +='\n'
        return nameStr


