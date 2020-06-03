import sqlite3
import os
import numpy as np
from obspy import UTCDateTime,read,read_inventory
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
        self.InventoryD={}
    def __call__(self,net,sta,comp,time0,time1):
        '''
        you should specify the timeL and staDirL by net sta comp and time0/1

        '''
        time1 = max(time0+1 ,time1)
        staDirL = self.getStaDirL(net,sta)
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
        elif net == 'YP':
            #staDir/2009/R304/NE67.2009.304.00.00.00.BHZ.sac
            pattern = '%s/%s/%s.%s.*%s.sac'%(staDir,time.strftime('%Y/R%j'),sta,\
                time.strftime('%Y.%j'),comp)
        #print('##',pattern)
        return glob(pattern)
    def getStaDirL(self,net,sta):
        staDirL = []
        if net == 'GS' or net =='NM':
            staDirL = ['/media/jiangyr/shanxidata21/nmSacData/'+net+\
            '.'+sta+'/']
        if net == 'YP':
            staDirL = ['/media/commonMount/data2/NECESSARRAY_SAC/NEdata*/%s/'%sta]
        staName = net + ' ' + sta
        if staName in self.himaDir:
            staDirL = self.himaDir[staName]
        return staDirL
    def getSensorDas(self,net,sta):
        staDirL = self.getStaDirL(net,sta)
        if net == 'YP':
            resp = glob('resp/YP/*%s*%s*Z'%(net,sta))
            if len(resp)>0:
                return resp[0], '130S','NECE'
            else:
                return 'UNKNOWN','UNKNOWN','UNKNOWN'
        if net == 'hima':
            logFileL=[]
            for staDir in staDirL:
                print(staDir+'*.log')
                logFileL += glob(staDir+'*.log')
            #print(logFileL)
            logFile = logFileL[0]
            sensorName = 'UNKNOWN'
            dasName    = 'UNKNOWN'
            sensorNum  = 'UNKNOWN'
            for logFile in logFileL:
                with open(logFile) as f:
                    while sensorName=='UNKNOWN'or dasName=='UNKNOWN'or sensorNum=='UNKNOWN':
                        line = f.readline()
                        if line=='':
                            break
                        if sensorName == 'UNKNOWN':
                            if 'Sensor Model' in line:
                                if len(line)<=21:
                                    tmp = 'UNKNOWN'
                                else:
                                    tmp = line[20:].split()
                                    if len(tmp)>0:
                                        tmp0 =tmp
                                        tmp = ''
                                        for TMP in tmp0:
                                            tmp+=TMP
                                    else:
                                        tmp = 'UNKNOWN'
                                sensorName = tmp
                                #if len(sensorName)<3:
                                #    sensorName= line
                                print(line)
                                continue
                        if sensorNum == 'UNKNOWN':
                            if 'Sensor Serial Number' in line:
                                if len(line)<=29:
                                    tmp = 'UNKNOWN'
                                else:
                                    tmp = line[28:].split()
                                    if len(tmp)>0:
                                        tmp0 =tmp
                                        tmp = ''
                                        for TMP in tmp0:
                                            tmp+=TMP
                                    else:
                                        tmp = 'UNKNOWN'
                                sensorNum = tmp
                                print(line)
                                continue
                        if dasName =='UNKNOWN':
                            if 'REF TEK' in line:
                                dasName    = line.split()[-1]
                                print(line)
                                continue
                if len(sensorName)!=0 and len(dasName)!=0:
                    return sensorName,dasName,sensorNum
                else:
                    return 'UNKNOWN','UNKNOWN','UNKNOWN'
    def getInventory(self,net,sta,sensorName='',dasName='',comp='BHZ'):
        respDir = 'resp/'
        if sensorName=='' or dasName=='':
            sensorName, dasName, sensorNum =self.getSensorDas(net,sta)
        if net =='YP':
            sensorName=sensorName[:-3]
        if sensorName+comp not in self.InventoryD:
            if net == 'hima':
                file='%s/%s.%s.resp'%(respDir,net,sensorName)
            if net == 'YP':
                file = sensorName+comp
            self.InventoryD[sensorName+comp] = read_inventory(file)
        if dasName+comp not in self.InventoryD:
            file = '%s/%s.%s.resp'%(respDir,net,dasName)
            self.InventoryD[dasName+comp]=\
            read_inventory(file)
        return self.InventoryD[sensorName+comp],self.InventoryD[dasName+comp]


#/media/jiangyr/shanxidata21/nmSacData/GS.HXP//201410//GS.HXP.20141001.BHZ.SAC