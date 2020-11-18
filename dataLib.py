import sqlite3
import os
import numpy as np
from obspy import UTCDateTime,read,read_inventory
from glob import glob
himaNet = ['hima']
NECE    = ['YP','AH','BJ','BU','CQ','FJ','GD','HA','HB','HE',\
'HI','HL','JL','JS','JX','LN','NM','NX','QH','SC','SD','SH','SN',\
'SX','TJ','XJ','XZ','YN','ZJ',]
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
    def __call__(self,net,sta,comp,time0,time1,nameMode=''):
        '''
        you should specify the timeL and staDirL by net sta comp and time0/1

        '''
        if nameMode == '':
            nameMode = net
        time1 = max(time0+1 ,time1)
        staDirL = self.getStaDirL(net,sta,nameMode)
        timeL = np.arange(time0,time1,3600).tolist()
        timeL.append(time1)
        fileL = []
        for staDir in staDirL:
            for time in timeL:
                tmpL = self.getFile(net,sta,comp,time,staDir,nameMode= nameMode)
                for tmp in tmpL:
                    if tmp not in fileL:
                        fileL.append(tmp)
        return fileL
    def getFile(self,net,sta,comp,time,staDir,nameMode=''):
        if not isinstance(time,UTCDateTime):
            time = UTCDateTime(time)
        if nameMode == '':
            nameMode = net
        if nameMode =='hima':
            pattern = '%s/%s.*.%s.m'%(staDir,time.strftime('R%j.01/%H/%y.%j')\
                ,filePath.himaComp[comp[-1]])
        elif nameMode in ['GS','NM']:
            pattern = '%s/%s/%s.%s.%s.%s.SAC'%(staDir,time.strftime('%Y%m/'),\
                net,sta,time.strftime('%Y%m%d'),comp)
        elif nameMode == 'YP':
            #staDir/2009/R304/NE67.2009.304.00.00.00.BHZ.sac
            pattern = '%s/%s/%s.%s.*%s.sac'%(staDir,time.strftime('%Y/R%j'),sta,\
                time.strftime('%Y.%j'),comp)
        elif nameMode == 'CEA':
            if True:
                time = time+8*3600
                pattern = '%s/%s/%s*%s.sac'%(staDir,time.strftime('%Y/%j'),sta,comp)
            else:
            #staDir/2009/R304/NE67.2009.304.00.00.00.BHZ.sac
            #2010/001/ANQ.2009365160003.AH.00.D.BHN.sac
                pattern = '%s/%s/%s*%s.sac'%(staDir,time.strftime('%Y/%j'),sta,comp)
        elif nameMode == 'CEAO':
            time = time+8*3600
            pattern = '%s/%s/%s*%s.sac'%(staDir,time.strftime('%Y%m%d/'),sta,comp)
        elif nameMode =='YNSC':
            pattern='%s/%s/%s/%s/%s.D/%s.%s.00.%s.D.%s.%s'\
            %(staDir,time.strftime('%Y'),net,sta,comp,net,sta,comp,\
                time.strftime('%Y'),time.strftime('%j'))
        elif nameMode =='XU':
            sta = sta.split('_')[0]
            pattern='%s/%s/%s*%s.sac'\
            %(staDir,sta,time.strftime('%Y/%Y%m%d/%Y%m%d'),comp)

        #print('##',pattern)
        return glob(pattern)
    def getStaDirL(self,net,sta,nameMode=''):
        staDirL = []

        if nameMode == '':
            nameMode = net
        if nameMode in ['GS','NM']:
            staDirL = ['/media/jiangyr/shanxidata21/nmSacData/'+net+\
            '.'+sta+'/']
        if nameMode == 'YP':
            staDirL = ['/media/commonMount/data2/NECESSARRAY_SAC/NEdata*/%s/'%sta]
        if nameMode == 'CEA':
            staDirL = ['/net/CEA/CEA1s/CEA/%s/%s/'%(net,sta),'/net/CEA/CEA1s/CEA_old/%s/%s/'%(net,sta)]
        staName = net + ' ' + sta
        if nameMode == 'CEAO':
            staDirL = ['/net/CEA/CEA_*/']
        if staName in self.himaDir:
            staDirL = self.himaDir[staName]
        if nameMode =='YNSC':
            if net =='YN':
                staDirL = ['/net/CEA/CEA0/net_yn/']
            elif net == 'SC':
                staDirL = ['/net/CEA/CEA1/net_sc/']

        if nameMode =='XU':
            staDirL = ['/HOME/jiangyr/YNSCMOVE/']
        return staDirL
    def getSensorDas(self,net,sta,nameMode=''):
        if nameMode == '':
            nameMode = net
        staDirL = self.getStaDirL(net,sta)
        if nameMode == 'YP':
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
    def getInventory(self,net,sta,sensorName='',dasName='',comp='BHZ',nameMode=''):
        if nameMode == '':
            nameMode = net
        respDir = 'resp/'
        if sensorName=='' or dasName=='':
            sensorName, dasName, sensorNum =self.getSensorDas(net,sta)
        if nameMode in ['YP','CEA']:
            sensorName=sensorName[:-3]
        if sensorName+comp not in self.InventoryD:
            if nameMode == 'hima':
                file='%s/%s.%s.resp'%(respDir,net,sensorName)
            if nameMode in ['YP','CEA']:
                file = sensorName+comp
            if '*' not in file and '?' not in file:
                self.InventoryD[sensorName+comp] = read_inventory(file)
            else:
                fileL = glob(file)
                self.InventoryD[sensorName+comp] = read_inventory(fileL[0])
                for fileTmp in fileL[1:]:
                    self.InventoryD[sensorName+comp] += read_inventory(fileTmp)
        if dasName+comp not in self.InventoryD:
            file = '%s/%s.%s.resp'%(respDir,net,dasName)
            self.InventoryD[dasName+comp]=\
            read_inventory(file)
        return self.InventoryD[sensorName+comp],self.InventoryD[dasName+comp]


#/media/jiangyr/shanxidata21/nmSacData/GS.HXP//201410//GS.HXP.20141001.BHZ.SAC