import numpy as np
import scipy
import matplotlib.pyplot as plt
import obspy
import os
defaultPath='fkRun/'
orignExe = '/Users/jiangyiran/prog/fk/fk/'
class FK:
    '''
    class for using fk program
    '''
    def __init__(self,exePath=defaultPath,orignExe=orignExe):
        self.exePath = exePath
        self.resDir = 'fkRes'
        self.tmpFile = ['res.%s'% s for s in 'ztr']
        self.orignExe = orignExe
        self.prepare()

    def prepare(self):
        if not os.path.exists(self.exePath):
            os.mkdir(self.exePath)
        if not os.path.exists(self.resDir):
            os.mkdir(self.resDir)
        exeFiles = ['fk.pl','syn','trav','fk2mt','st_fk','fk','hk']
        for exeFile in exeFiles:
            os.system('cp %s %s'%(self.orignExe+exeFile,self.exePath))

    def clear(self,clearRes = False):
        os.system('rm -r %s'%self.exePath)
        if clearRes:
            os.system('rm -r %s'%self.resDir)
        self.prepare()

    def calGreen(self,distance=[1],modelFile='paper',fok='/k',srcType=[0,2],rdep=0,\
        isDeg = False, dt=1, expnt=8, expsmth = 0,f=[0,0],p=[0,0],kmax=0,\
        updn=0,depth=1, taper=0.3, dk=0.3,cmd=''):
        if fok =='/f' or fok == '':
            modelFile+='fk1'
        else:
            modelFile+='fk0'
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
            synCmd+=' -D%.2f/%.2f '%(dura,rise)
            if len(srcSac)>0:
                synCmd +=' -S%s '%srcSac
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
        for dis in self.distance:
            for azi in self.azimuth:
                sacNames = self.getFileName(dis,self.depth,azi,self.M)
                sacs = [obspy.read(sacName)[0] for sacName in sacNames]
                sacsL.append(sacs)
        return sacsL
    def test(self,distance=[50],modelFile='hk',fok='/k',dt=1,depth=15,\
        expnt=10,dura=10,dk=-1,azimuth=[0],M=[3e25,0,2,3,0,1,0]):
        self.calGreen(distance=distance,modelFile=modelFile,fok=fok,dt=dt,depth=depth,expnt=expnt,dk=dk)
        self.syn(dura=dura,azimuth=azimuth,M=M)
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


