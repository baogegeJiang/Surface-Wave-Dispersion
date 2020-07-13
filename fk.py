import numpy as np
import scipy
import matplotlib.pyplot as plt
import obspy
import os
from multiprocessing import Process, Manager
import mathFunc
defaultPath='fkRun/'
orignExe = '/Users/jiangyiran/prog/fk/fk/'
###different source different meaning
class FK:
    '''
    class for using fk program
    '''
    def __init__(self,exePath=defaultPath,orignExe=orignExe,resDir='fkRes'):
        self.exePath = exePath
        self.resDir = resDir
        self.tmpFile = ['res.%s'% s for s in 'ztr']
        self.orignExe = orignExe
        self.prepare()

    def prepare(self):
        if not os.path.exists(self.exePath):
            os.makedirs(self.exePath)
        if not os.path.exists(self.resDir):
            os.makedirs(self.resDir)
        exeFiles = ['fk.pl','syn','trav','fk2mt','st_fk','fk','hk']
        for exeFile in exeFiles:
            if not os.path.exists(self.exePath+'/'+exeFile):
                os.system('cp %s %s'%(self.orignExe+exeFile,self.exePath))

    def clear(self,clearRes = False):
        os.system('rm -r %s'%self.exePath)
        if clearRes:
            os.system('rm -r %s'%self.resDir)
        self.prepare()

    def calGreen(self,distance=[1],modelFile='paper',fok='/k',srcType=[0,2],rdep=0,\
        isDeg = False, dt=1, expnt=8, expsmth = 0,f=[0,0],p=[0,0],kmax=0,\
        updn=0,depth=1, taper=0.3, dk=0.3,cmd='',isFlat=False):
        if fok =='/f' or fok == '':
            modelFile+='fk1'
        else:
            modelFile+='fk0'
        if isFlat:
            modelFile +='_flat'
        if fok == '/f' and isFlat:
            print('warning, may do flat twice')
        if not os.path.exists(modelFile):
            modelFile = modelFile[:-1]
        copyModelCmd = 'cp %s %s'%(modelFile,self.exePath)
        os.system(copyModelCmd)
        baseModelName = os.path.basename(modelFile)
        greenCmd = 'cd %s;./fk.pl '%self.exePath
        greenCmd+=' -M%s/%d%s '%(baseModelName,depth,fok)
        if isDeg:
            greenCmd+=' -D '
        if f[0]>0:
            greenCmd+=' -H %.5f/%.5f '%(f[0], f[1])
        greenCmd +=' -N%d/%.3f'%(2**expnt,dt)
        if expsmth >=0:
            greenCmd+='/%d'%(2**expsmth)
        if dk>0:
            greenCmd+='/%f'%dk
            if taper>0:
                greenCmd+='/%.3f'%taper
        greenCmd+=' '
        if p[0]>0:
            greenCmd +=' -P%.3f/%.3f'%(p[0],p[1])
            if kmax > 0:
                greenCmd += '/%.3f'%kmax
            greenCmd+=' '
        if rdep!=0:
            greenCmd +=' -R%d '%rdep
        #greenCmd +=' -S%d '%srcType
        greenCmd += ' -U%d '%updn
        if len(cmd)>0:
            greenCmd +=' -X%s '%cmd
        greenCmd0 = greenCmd
        for src in srcType:
            greenCmd = greenCmd0
            greenCmd +=' -S%d '%src
            for dis in distance:
                greenCmd += ' %d'%dis
            print(greenCmd)
            os.system(greenCmd)
        self.greenRes = baseModelName + '_%d'%depth
        self.depth = depth
        self.distance = distance
        self.rdep     = rdep
        self.modelFile = modelFile
    def syn(self,M=[3e25,0,2,3,0,1,0],azimuth=[0],dura=1,rise=0.2,srcSac='',f=[0,0],\
        ):
        self.azimuth = azimuth
        for azi in self.azimuth:
            synCmd = 'cd %s/;./syn -M%.5f'%(self.exePath,M[0])
            self.M =M
            self.source = '%.2f_%.2f'%(dura,rise)
            for m in M[1:]:
                synCmd += '/%.5f'%m
            synCmd+=' '
            synCmd+=' -A%.2f '%azi
            
            if len(srcSac)>0:
                synCmd +=' -S%s '%srcSac
            else:
                synCmd+=' -D%.2f/%.2f '%(dura,rise)
            if f[0]>0:
                synCmd+=' -F%.5f/%.5f '%(f[0],f[1])
            for dis in self.distance:
                #1.0000_0.000.grn.0
                firstFile = '%s/%d.grn.0'%(self.greenRes,dis)
                tmpCmd = synCmd
                tmpCmd += ' -G%s '%firstFile
                tmpCmd += ' -O%s'%self.tmpFile[0]
                print(tmpCmd)
                os.system(tmpCmd)
                self.mvSac(dis,azi)
    def mvSac(self, dis,azi):
        fileName = self.getFileName(dis,self.depth,azi,self.M)
        for i in range(3):
            resDir =os.path.dirname(fileName[i])
            if not os.path.exists(resDir):
                os.makedirs(resDir)
            mvCmd = 'mv %s %s'%(self.exePath+self.tmpFile[i],fileName[i])
            os.system(mvCmd)
    def getFileName(self, dis,depth, azimuth, M):
        dirName = '%s/%s/%d/'%(self.resDir,os.path.basename(self.modelFile),depth)
        basename ='%s_%d_%.2f'%(self.source,dis,azimuth)
        for m in M:
            basename+='%.3f_'%m
        return [(dirName+basename[:-1]+'.%s')%s for s in 'ztr']
    def readAll(self):
        sacsL=[]
        sacNamesL=[]
        for dis in self.distance:
            for azi in self.azimuth:
                sacNames = self.getFileName(dis,self.depth,azi,self.M)
                sacs = [obspy.read(sacName)[0] for sacName in sacNames]
                for sacName in sacNames:
                    trace = obspy.read(sacName)[0]
                    trace.stats['sac']['kstnm']=str(dis)[:3]+str(azi)[:3]
                    trace.write(sacName,format="SAC")
                sacNamesL.append(sacNames)
                sacsL.append(sacs)
        return sacsL,sacNamesL
    def test(self,distance=[50],modelFile='hk',fok='/k',dt=1,depth=15,\
        expnt=10,dura=10,dk=-1,azimuth=[0],M=[3e25,0,2,3,0,1,0],isFlat=False,\
        srcSac='',rise=0.2):
        self.calGreen(distance=distance,modelFile=modelFile,fok=fok,dt=dt,\
            depth=depth,expnt=expnt,dk=dk,isFlat=isFlat)
        self.syn(dura=dura,azimuth=azimuth,M=M,srcSac=srcSac,rise=rise)
        sacsL = self.readAll()
        '''
        for sacs  in sacsL:
            for sac in sacs:
                sac.plot()
        '''
        self.sacsL = sacsL
        return sacsL
    def dispersion(self,sac):
        data = sac.data
        delta = sac.stats['sac']['delta']
        fs = 1/delta
        F,t,zxx = scipy.signal.stft(data,fs=fs,nperseg=100,noverlap=98)
        return F, t,zxx
    def testDis(self):
        for sacs in self.sacsL:
            for sac in sacs:
                F,t,zxx =self.dispersion(sac)
                plt.subplot(2,1,1);plt.pcolor(t,F,np.abs(zxx));plt.subplot(2,1,2);plt.plot(sac.data);plt.show()
    def genSourceSac(self,filename,data=np.random.rand(100),b=0,delta=1,t=obspy.UTCDateTime(0)):
        header = {'kstnm': 'FK', 'kcmpnm': 'BHZ', 'stla': 0,\
        'stlo': 0,'evla': 0, 'evlo': 0, 'evdp': 0,\
        'nzyear': t.year,'nzjday': t.julday, 'nzhour': t.hour, \
        'nzmin': t.minute,'nzsec': t.second,'nzmsec': int(t.microsecond/1e3)\
        , 'delta': delta,'b':b}
        sac = obspy.io.sac.sactrace.SACTrace(data=data,**header).to_obspy_trace()
        sac.write(filename)
    def genSourceSacs(self,fileNameL,delta=0.5,time=50):
        count = int(time/delta)
        i=0
        for file in fileNameL:
            #print(file)
            dura = 8+40*np.random.rand()+i%12
            duraCount = int(dura/delta)
            data = np.zeros(count)
            mathFunc.randomSource(i%4,duraCount,data)
            data/=data.sum()
            i+=1
            self.genSourceSac(file,data,delta=delta)

def FKL(n,exePath='FKRUN/',orignExe=orignExe,resDir='FKRES/'):
    return fkL(n,exePath='FKRUN/',orignExe=orignExe,resDir='FKRES/').fL

class fkL(list) : 
    def __init__(self,n,exePath='FKRUN/',orignExe=orignExe,resDir='FKRES/'):
        super(fkL, self).__init__()
        for i in range(n):
            self.append(FK(exePath=exePath+'/%d/'%i,orignExe=orignExe,resDir=resDir+'/%d/'%i))
    def __call__(self,num,target):
        fkN = len(self)
        pL = []
        for i in range(fkN):
            pL.append(Process(\
                target=target,\
                args=(i,range(i,num,fkN),self[i]) 
                )\
            )
            pL[-1].start()
        for p in pL:
            p.join()
            print('######',i)
            i+=1
    def clear(self):
        for f in self:
            f.clear()

def genSourceSacs(f,N,delta,srcSacDir = \
    '/home/jiangyr/Surface-Wave-Dispersion/srcSac/',time=50):
    fileNameL = [getSourceSacName(index,delta,srcSacDir) for index in range(N)]
    #print(fileNameL)
    if not os.path.exists(srcSacDir):
        os.makedirs(srcSacDir)
    f.genSourceSacs(fileNameL,delta,time=time)

def getSourceSacName(index,delta,srcSacDir = \
    '/home/jiangyr/Surface-Wave-Dispersion/srcSac/'):
    if index<0:
        return ''
    return '%s/%d_%d.sac'%(srcSacDir,index,delta*1e3)

        


