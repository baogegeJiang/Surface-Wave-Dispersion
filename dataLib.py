import sqlite3
import os
import numpy as np
from obspy import UTCDateTime,read
from glob import glob
def convertFileLst(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
    D = {}
    for line in lines:
        elements = line.split()
        staName = 'hima '+elements[0]
        if staName not in D:
            D[staName] = []
        tmpDir = elements[-1][:-7]
        if tmpDir not in D[staName]:
            D[staName].append(tmpDir)
    with open(fileName+'_New','w') as f:
        for staName in D:
            f.write(staName)
            for tmpDir in D[staName]:
                f.write(' '+tmpDir)
            f.write('\n')

class filePath:
    himaComp= {'E':'3','N':'2','Z':'1'}
    def __init__(self,himaFile='fileLst_New'):
        self.himaDir = {}
        if os.path.exists(himaFile):
            with open(himaFile, 'r') as f:
                lines = f.readlines()
            for line in lines:
                elements = line.split()
                staName = elements[0] + ' ' +elements[1]
                self.himaDir[staName] = elements[2:]
    def __call__(self,net,sta,comp,time0,time1):
        time1 = max(time0+1 ,time1)
        staDirL = []
        if net == 'GS' or net =='NM':
            staDirL = ['/media/jiangyr/shanxidata21/nmSacData/'+net+\
            '.'+sta+'/']
        staName = net + ' ' + sta
        if staName in self.himaDir:
            staDirL = self.himaDir[staName]
        timeL = np.arange(time0,time1,3600).tolist()
        timeL.append(time1)
        fileL = []
        for staDir in staDirL:
            for time in timeL:
                tmpL = self.getFile(net,sta,comp,time,staDir)
                for tmp in tmpL:
                    if tmp not in fileL:
                        fileL.append(tmp)
        return fileL
    def getFile(self,net,sta,comp,time,staDir):
        if not isinstance(time,UTCDateTime):
            time = UTCDateTime(time)
        if net == 'hima':
            pattern = '%s/%s.*.%s.m'%(staDir,time.strftime('R%j.01/%H/%y.%j')\
                ,filePath.himaComp[comp[-1]])
        elif net == 'GS' or net == 'NM':
            pattern = '%s/%s/%s.%s.%s.%s.SAC'%(staDir,time.strftime('%Y%m/'),\
                net,sta,time.strftime('%Y%m%d'),comp)
        #print(pattern)
        return glob(pattern)


#/media/jiangyr/shanxidata21/nmSacData/GS.HXP//201410//GS.HXP.20141001.BHZ.SAC