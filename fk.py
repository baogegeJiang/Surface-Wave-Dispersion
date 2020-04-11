import numpy as np
import scipy
import matplotlib.pyplot as plt
import obspy
import os
defaultPath='fkRun/'
orignExe = '/Users/jiangyiran/prog/fk/fk/'
class FK:
    def __init__(self,exePath=defaultPath):
        self.exePath = exePath
        self.resDir = 'fkRes'
        self.tmpFile = ['res.%s'% s for s in 'ztr']
        self.prepare()

    def prepare(self):
        if not os.path.exists(self.exePath):
            os.mkdir(self.exePath)
        if not os.path.exists(self.resDir):
            os.mkdir(self.resDir)
        exeFiles = ['fk.pl','syn','trav','fk2mt','st_fk','fk']
        for exeFile in exeFiles:
            os.system('cp %s %s'%(orignExe+exeFile,self.exePath))

    def clear(self,clearRes = False):
        os.rmdir(self.exePath)
        if clearRes:
            os.rmdir(self.resDir)
        self.prepare()

    def calGreen(self,distance=[1],modelFile='paperfk0',fok='',srcType=[0,2],rdep=0,\
        isDeg = False, dt=1, expnt=8, expsmth = 0,f=[0,0],p=[0,0],kmax=0,\
        updn=0,depth=1, taper=-1, dk=-1,cmd=''):
        copyModelCmd = 'cp %s %s'%(modelFile,self.exePath)
        os.system(copyModelCmd)
        baseModelName = os.path.basename(modelFile)
        greenCmd = 'cd %s;./fk.pl '%self.exePath
        greenCmd+=' -M%s/%d%s '%(modelFile,depth,fok)
        if isDeg:
            greenCmd+=' -D '
        if f[0]>0:
            greenCmd+=' -H %.5f/%.5f '%(f[0], f[1])
        greenCmd +=' -N%d/%.3f'%(2**expnt,dt)
        if expsmth !=0:
            greenCmd+='%d'%(2**expsmth)
        if dk>0:
            greenCmd+='/%.3f'%dk
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
        self.greenRes = modelFile + '_%d'%depth
        self.distance = distance
        self.rdep     = rdep
        self.modelFile = modelFile
    def syn(self,M=[1,0,2,3,0,1,0],azimuth=0,dura=1,rise=0.2,srcSac='',f=[0,0],\
        ):
        synCmd = 'cd %s/;./syn -M%.5f'%(self.exePath,M[0])
        for m in M[1:]:
            synCmd += '/%.5f'%m
        synCmd+=' '
        synCmd+=' -A%.2f '%azimuth
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
    def mvSac(self, dis, azimuth, M):
        fileName = self.getFileName(dis,azimuth,M)
        for i in range(3):
            resDir =os.path.dirname(fileName[i])
            if not os.path.exists(resDir):
                os.mkdir(resDir)
            mvCmd = 'mv %s %s'%(self.exePath+self.tmpFile[i],fileName[i])
    def getFileName(self, dis, azimuth, M):
        dirName = '%s/%s/%d/'%(self.resDir,self.modelFile,dis)
        basename ='%d_'%azimuth
        for m in M:
            basename+='%.3f_'%m
        return [(dirName+basename[-1]+'.%s')%s for s in 'ztr']
    def test(self):
        self.calGreen()
        self.syn()

