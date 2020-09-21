import obspy 
import numpy as np
from distaz import DistAz
from dataLib import filePath
from obspy import UTCDateTime,read
from distaz import DistAz
import os 
import random
from matplotlib import pyplot as plt
import time
fileP = filePath()
def tolist(s,d='/'):
    return s.split(d)
nickStrL='1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
strType={'S':str,'f':float,'F':float,'i':int, 'l':tolist,'b':bool}
NoneType = type(None)


class Dist:
    '''
    class which can be easily extended for different object with 
    accesses to inputing and outputing
    keys for the 
    '''
    def __init__(self,*argv,**kwargs):
        self.defaultSet()
        self.splitKey = ' '
        self.l = [None for i in range(len(self.keys))]
        for i in range(len(self.keys0)):
            self.l[i] = self.keys0[i]
        if 'keysIn' in kwargs :
            if isinstance(kwargs['keysIn'],list):
                self.keysIn     = kwargs['keysIn']
            else:
                self.keysIn     = kwargs['keysIn'].split()
        for i in range(len(argv)):
            self[self.keysIn[i]] = argv[i]
        if 'splitKey' in kwargs:
            self.splitKey = kwargs['splitKey']

        if 'line' in kwargs:
            self.setByLine(kwargs['line'])
        
        #print(kwargs)
        for key in self.keys:
            if key in kwargs:
                self[key]=kwargs[key]
    def defaultSet(self):
        self.keys = ['']
        self.keysType =['s']
        self.keysIn = ['']
        self.keysName = ['']
        self.keys0 = [None]
    def index(self,key):
        if key in self.keys:
            return self.keys.index(key)
        return -1
    def __getitem__(self,key):
        if not key in self.keys:
            print('no ',key)
        return self.l[self.index(key)]
    def __setitem__(self,key,value):
        if not key in self.keys:
            print('no ',key)
        self.l[self.index(key)] = value

    def setByLine(self, line):
        if self.splitKey != ' ':
            tmp = line.split(self.splitKey)
        else:
            tmp = line.split()
        #print(tmp,self.splitKey)
        for i in range(min(len(tmp),len(self.keysIn))):
            tmp[i] = tmp[i].strip()
            index = self.index(self.keysIn[i])
            if tmp[i]!='-99999':
                #print(self.keysIn[i],index,self.keysType[index][0])
                self[self.keysIn[i]] = strType[self.keysType[index][0]](tmp[i])
            else:
                self[self.keysIn[i]] = None
    def __str__(self,*argv):
        line = ''
        if len(argv)>0:
            keysOut = argv[0]
        else:
            keysOut =self.keysIn
        s= self.splitKey
        if len(argv)>1:
            s = argv[1]
        for key in keysOut:
            if not isinstance(self[key],type(None)):
                line += str(self[key])+s
            else:
                line += '-99999 '
        return line[:-1]
    def __repr__(self):
        return self.__str__()
    def __iter__(self):
        return self.keys.__iter__()
    def copy(self):
        type(self)()
        inD = {'keysIn':self.keys}
        for key in self:
            inD[key]  = self[key]
        selfNew=type(self)(**inD)
        selfNew.keysIn = self.keysIn.copy()
        selfNew.splitKey = self.splitKey
        return selfNew
    def update(self,selfNew):
        for key in selfNew:
            if not isinstance(selfNew[key],NoneType):
                self[key] = selfNew[key]
    def keyIn(self):
        keyIn = ''
        for tmp in self.keysIn:
            keyIn += tmp  + ' ' 
        return keyIn[:-1]
    def name(self,s =' '):
        return self.__str__(self.keysName,s)
    def __eq__(self,name1):
        name0 = ''
        if  isinstance(name1,type(self)):
            self1 = name1
            name1 = ''
            for key in self1.keysName:
                name1 = self1.name()
        for key in self.keysName:
            name0 = self.name()
        return name0 == name1
    def loc(self):
        return [self['la'],self['lo'],self['dep']]
    def distaz(self,loc):
        if isinstance(loc,list) or isinstance(loc,np.ndarray):
            dis = DistAz(self['la'],self['lo'],loc[0],loc[1])
        else:
            dis = DistAz(self['la'],self['lo'],loc['la'],loc['lo'])
        return dis
    def dist(self,loc):
        dis = self.distaz(loc)
        return dis.degreesToKilometers(dis.getDelta())
    def az(self,loc):
        dis = self.distaz(loc)
        return dis.getAz()
    def baz(self,loc):
        dis = self.distaz(loc)
        return dis.getBaz()

defaultStats = {'network':'00','station':'00000','channel':'000'}
class Station(Dist):
    def __init__(self,*argv,**kwargs):
        super().__init__(*argv,**kwargs)
        #if not isinstance(self['compBase'],NoneType): 
        #    self['comp'] = [self['compBase']+s for s in 'ENZ' ]
        #print(self['index'])
        if isinstance( self['index'],NoneType)==False and isinstance( self['nickName'],NoneType):
            self['nickName'] = self.getNickName(self['index'])
        self.defaultStats = defaultStats
    def defaultSet(self):
        super().defaultSet()
        self.keysIn   = 'net sta compBase lo la erroLo erroLa dep erroDep '.split()
        self.keys     = 'net sta compBase lo la erroLo erroLa dep erroDep nickName comp index nameFunc sensorName dasName sensorNum nameMode netSta doFilt oRemove'.split()
        self.keysType ='S S S f f f f f f S l f F S S S S S b b'.split()
        self.keys0 =    [None,None,'BH',None,None,0    ,   0, 0   ,0,  None,     None,None, fileP,'','','','','',True,False]
        self.keysName = ['net','sta']
    def getNickName(self, index):
        nickName = ''
        N      = len(nickStrL)
        for i in range(4):
            tmp    = index%N
            nickName += nickStrL[tmp]
            count  = int(index/N)
        return nickName
    def getFileNames(self, time0,time1=None):
        if isinstance(time1, NoneType):
            time1 = time0+86399
        return [self['nameFunc'](self['net'],self['sta'], \
            comp, time0,time1,self['nameMode']) for comp in self['comp']]
    def __setitem__(self,key,value):
        super().__setitem__(key,value)
        if not key in self.keys:
            print('no ',key)
        self.l[self.index(key)] = value
        if key =='compBase':
            self['comp'] = [self['compBase']+s for s in 'ENZ']

        if key =='net' and self['nameMode']=='':
            self['nameMode'] = self['net']
        if key =='netSta' and self[key]!='' and isinstance(self['net'],NoneType)\
         and isinstance(self['sta'],NoneType):
            self['net'], self['sta'] = self[key].split('.')[:2]
    def baseSacName(self,resDir='',strL='ENZ',infoStr=''):
        if infoStr=='':
            return [ resDir+'/'+self['net']+'.'+self['sta']+'.'+self['compBase']+comp for comp in strL]
        else:
            return [ resDir+'/'+self['net']+'.'+self['sta']+'.'+infoStr+'.'+self['compBase']+comp for comp in strL]
    def set(self,key,value):
        for tmp in self:
            tmp[key] = value
    def getInventory(self):
        self.sensor=[]
        self.das=[]
        for i in range(3):
            sensor, das=  self['nameFunc'].getInventory(self['net'],self['sta'],\
                self['sensorName'],self['dasName'],comp=self['comp'][i],nameMode=self['nameMode'])
            self.sensor.append(sensor)
            self.das.append(das)
        return self.sensor,self.das
    def getSensorDas(self):
        if self['sensorName'] == '' or self['dasName']=='':
            sensorName,dasName,sensorNum=self['nameFunc'].getSensorDas(self['net'],self['sta'],nameMode=self['nameMode'])
            self['sensorName'] = sensorName
            self['dasName']    = dasName
            self['sensorNum'] = sensorNum
            if self['net'] == 'YP':
                self['compBase']=self['sensorName'][-3:-1]
                self['comp'] = [self['compBase']+s for s in 'ENZ' ]
        return self['sensorName'],self['dasName'],self['sensorNum']

class StationList(list):
    def __init__(self,*argv,**kwargs):
        super().__init__()
        self.inD = {}
        if isinstance(argv[0],list):
            for sta in argv[0]:
                self.append(sta)
        if isinstance(argv[0],str):
            self.read(argv[0])
    def __add__(self,self1):
        selfNew = StationList([])
        for station in self:
            selfNew.append(station)
        for station in self1:
            selfNew.append(station)
        return selfNew
    def read(self,fileName):
        self.header = []
        with open(fileName,'r') as staFile:
            lines = staFile.readlines()
        inD = {}
        if lines[0][0]=='#':
            keys = lines[0][1:]
            lines= lines[1:]
            inD['keysIn'] = keys
        index=0
        for line in lines:
            inD['line'] = line
            inD['index']= index
            self.append(Station(**inD))
            index+=1
        self.inD = inD
    def write(self,fileName,*argv,**kwargs):
        with open(fileName,'w+') as f:
            keysOut = ''
            if 'keysIn' in self.inD:
                keysOut = self.inD['keysIn']
            if len(argv)>0:
                keysOut = argv[0]
            if len(keysOut) >0:
                keysOut = keysOut.split()
                for sta in self:
                    sta.keysIn = keysOut
            keysOut = '#' + self[0].keyIn()+'\n'
            f.write(keysOut)
            if 'indexL' in kwargs:
                f.write(self.__str__(kwargs['indexL']))
            else:
                f.write(self.__str__())
    def __str__(self,indexL=[]):
        line =''
        if len(indexL)==0:
            for sta in self:
                line += '%s\n'%sta
        else:
            for i in indexL:
                sta = self[i]
                line += '%s\n'%sta
        return line 
    def loc0(self):
        loc = np.zeros(3)
        count = 0
        strL = ['la','lo','dep']
        for station in self:
            for i in range(3):
                tmpStr = strL[i]
                #print(station[tmpStr],i)
                if station[tmpStr] !=None:
                    loc[i] = loc[i] + station[tmpStr]
        return loc/len(self)
    def getInventory(self):
        for station in self:
            sensorName, dasName, sensorNum =  station.getSensorDas()
            if sensorName != 'UNKNOWN' and dasName != 'UNKNOWN':
                sensor,das=station.getInventory()
    def getSensorDas(self):
        for station in self:
            sensorName, dasName,sensorNum =  station.getSensorDas()
    def find(self,sta,net=''):
        for station in self:
            if station['sta'] != sta:
                continue
            if net !='' and station['net'] != net:
                continue
            return station
        return None
    def Find(self,netSta, spl='.'):
        netSta = netSta.split('/')[-1]
        net, sta = netSta.split(spl)[:2]
        return self.find(sta,net)
    def set(self,key,value):
        for tmp in self:
            tmp[key] = value


        
class Record(Dist):
    def __init__(self,*argv,**kwargs):
        super().__init__(*argv,**kwargs)
    def defaultSet(self):
        super().defaultSet()
        self.keysIn   = 'staIndex pTime sTime pProb sProb'.split()
        self.keys     = 'staIndex pTime sTime pProb sProb pCC  sCC  pM   pS   sM   sS staName no'.split()
        self.keysType = 'i        f     f     f      f    f    f    f    f    f    f  S'.split()
        self.keys0    = [0,       None,  None,None,  None,None,None,None,None,None,None,None]
        self.keysName = ['staIndex','pTime','sTime']
    def select(self,req):
        return True

defaultStrL='ENZ'
class Quake(Dist):
    def __init__(self,*argv,**kwargs):
        super().__init__(*argv,**kwargs)
        self.records = []
        if not isinstance(self['strTime'],NoneType):
            #print('**',self['strTime'])
            self['time'] = UTCDateTime(self['strTime']).timestamp
    def defaultSet(self):
        #               quake: 34.718277 105.928949 1388535219.080064 num: 7 index: 0    randID: 1    filename: 16071/1388535216_1.mat -0.300000
        super().defaultSet()
        self.keysIn   = 'type   la       lo          time          para0 num para1 index para2 randID para3 filename ml   dep'.split()
        self.keys     = 'type   la       lo          time          para0 num para1 index para2 randID para3 filename ml   dep stationList strTime no YMD HMS'.split()
        self.keysType = 'S      f        f           f             S     F   S     f     S     f      S     S        f    f   l  S S S S'.split()
        self.keys0    = [None,  None,     None,      None,         None, None,None,None, None, None,  None,  None,   None,0 ,'','']
        self.keysName = ['time','la','lo']
    def Append(self,tmp):
        if isinstance(tmp,Record):
            self.records.append(tmp)
        else:
            print('please pass in Record type')
    def calCover(self,stationList=[],maxDT=None):
        if len(stationList) ==0:
            stationList = self['stationList']
        if isinstance(stationList,type(None)) or len(stationList)==0:
            print('no stationInfo')
            return None
        coverL=np.zeros(360)
        for record in self.records:
            if record['pTime']==0 and record['sTime']==0:
                continue
            if maxDT!=None:
                if record['pTime']-self['time']>maxDT:
                    continue
            staIndex= int(record['index'])
            la      = stationList[staIndex]['la']
            lo      = stationList[staIndex]['lo']
            dep     = stationList[staIndex]['dep']/1e3
            delta,dk,Az = self.calDelta(la,lo,dep)
            R=int(60/(1+dk/200)+60)
            N=((int(Az)+np.arange(-R,R))%360).astype(np.int64)
            coverL[N]=coverL[N]+1
        L=((np.arange(360)+180)%360).astype(np.int64)
        coverL=np.sign(coverL)*np.sign(coverL[L])*(coverL+coverL[L])
        coverRate=np.sign(coverL).sum()/360
        return coverRate
    def calDelta(self,la,lo,dep=0):
        D=DistAz(la,lo,self['la'],self['lo'])
        delta=D.getDelta()
        dk=D.degreesToKilometers(delta)
        dk=np.linalg.norm(np.array([dk,self['dep']+dep]))
        Az=D.getAz()
        return delta,dk,Az
    def num(self):
        return len(self.records)
    def __str__(self, *argv):
        self['num'] = self.num()
        return super().__str__(*argv )
    def staIndexs(self):
        return [record['staIndex'] for record in self.records]
    def resDir(self,resDir):
        return '%s/%s/'%(resDir,self.name(s='_'))
    def cutSac(self, stations,bTime=-100,eTime=3000,resDir = 'eventSac/',para={},byRecord=True,isSkip=False):
        time0  = self['time'] + bTime
        time1  = self['time'] + eTime
        tmpDir = self.resDir(resDir)
        if not os.path.exists(tmpDir):
            os.makedirs(tmpDir)
        staIndexs = self.staIndexs()
        for staIndex in range(len(stations)):
            station = stations[staIndex]
            if len(staIndexs) >0 and staIndex not in staIndexs and byRecord:
                continue
            if staIndex in staIndexs:
                record = self.records[staIndexs.index(staIndex)]
            resSacNames = station.baseSacName(tmpDir)
            isF = True
            for resSacName in resSacNames:
                if not os.path.exists(resSacName):
                    isF = False
            if isF and isSkip:
                print(resSacNames,'done')
                continue
            sacsL = station.getFileNames(time0,time1+2)
            for i in range(3):
                sacs = sacsL[i]
                if len(sacs) ==0:
                    continue
                data = mergeSacByName(sacs, **para)
                if isinstance(data,NoneType):
                    continue
                data.data -= data.data.mean()
                data.detrend()
                #print(data)
                if data.stats.starttime<=time0 and data.stats.endtime >= time1:
                    data=data.slice(starttime=UTCDateTime(time0), \
                        endtime=UTCDateTime(time1), nearest_sample=True)
                    #print('#',data.stats.starttime.timestamp-time0)
                    #print('##',data.stats.endtime.timestamp -time1)
                    #print('###',time0 -time1)
                    #print('####',data.stats.starttime.timestamp-data.stats.endtime.timestamp)
                    decMul=-1
                    if 'delta0' in para:
                        decMul = para['delta0']/data.stats.delta
                        if np.abs(int(decMul)-decMul)<0.001:
                            decMul=decMul
                            #print('decMul: ',decMul)
                        else:
                            decMul=-1
                    data=adjust(data,decMul=decMul,stloc=station.loc(),eloc = self.loc(),\
                        kzTime=self['time'],sta = station['sta'],net=station['net'])
                    data.write(resSacNames[i],format='SAC')
        return None
    def getSacFiles(self,stations,isRead=False,resDir = 'eventSac/',strL='ENZ',\
        byRecord=True,maxDist=-1,minDist=-1,remove_resp=False,isPlot=False,\
        isSave=False,respStr='_remove_resp',para={},isSkip=False):
        sacsL = []
        staIndexs = self.staIndexs()
        tmpDir = self.resDir(resDir)
        para0 ={\
        'delta0'    :0.02,\
        'freq'      :[-1, -1],\
        'filterName':'bandpass',\
        'corners'   :2,\
        'zerophase' :True,\
        'maxA'      :1e5,\
        'output': 'VEL',\
        'toDisp': False
        }
        para0.update(para)
        print(para0)
        para = para0
        respStr += '_'+para['output']
        respDone = False
        if para['freq'][0] > 0:
            print('filting ',para['freq'])
        if 'pre_filt' in para:
            print('pre filting ',para['pre_filt'])
        if 'output' in para:
            print('outputing ',para['output'])
        for staIndex in range(len(stations)):
            station = stations[staIndex]
            if remove_resp and station['oRemove'] ==False:
                sensorName,dasName,sensorNum = station.getSensorDas()
                if sensorName=='UNKNOWN' or sensorName==''  \
                or dasName=='UNKNOWN' or dasName=='':
                    continue
                if station['net']=='YP' or station['nameMode'] == 'CEA':
                    rigthResp = True
                    for channelIndex in range(3):
                        #channelIndexO = defaultStrL.index(strL[channelIndex])
                        if station.sensor[channelIndex][0].code != station['net'] or \
                            station.sensor[channelIndex][0][0].code != station['sta'] or \
                            station.sensor[channelIndex][0][0][0].code != station['comp'][channelIndex]:
                            print('no Such Resp')
                            print(station.sensor[channelIndex][0][0][0].code,station['comp'][channelIndex])
                            #time.sleep(1)
                            rigthResp=False
                    if rigthResp == False:
                        print('#### no Such Resp')
                        continue
                if self['time']<UTCDateTime(2009,7,1).timestamp and station['nameMode']=='CEA'\
                    and len(station.sensor[0])<2:
                    continue
            if len(staIndexs) > 0 and staIndex not in staIndexs and byRecord:
                continue
            if staIndex in staIndexs:
                record = self.records[staIndexs.index(staIndex)]
            if maxDist>0 and self.dist(station)>maxDist:
                continue
            if minDist>0 and self.dist(station)<minDist:
                continue
            resSacNames = station.baseSacName(tmpDir,strL=strL)
            respDone = False
            if remove_resp and isSave==False and station['oRemove'] ==False:
                isF = True
                resSacNames = station.baseSacName(tmpDir,strL=strL,infoStr=respStr)
                for resSacName in resSacNames:
                    if not os.path.exists(resSacName):
                        isF = False
                        break
                if isF:
                    respDone = True
                    #print(station,'done')
                else:
                    resSacNames = station.baseSacName(tmpDir,strL=strL)
            if remove_resp and isSkip==True and isSave==True and station['oRemove'] ==False:
                isF = True
                resSacNamesTmp = station.baseSacName(tmpDir,strL=strL,infoStr=respStr)
                for resSacName in resSacNamesTmp:
                    if not os.path.exists(resSacName):
                        isF = False
                        #print('need do')
                        break
                if isF:
                    #print(station,'done')
                    continue   
            isF = True
            for resSacName in resSacNames:
                if not os.path.exists(resSacName):
                    isF = False
                    #print('no origin sac')
                    break
            if isF == True:
                if remove_resp and respDone == False and station['oRemove'] ==False and isSave == False \
                and isSkip:
                    continue
                if isRead:
                    #print(resSacNames)
                    sacsL.append([ obspy.read(resSacName)[0] for resSacName in resSacNames])
                    if isPlot:
                        plt.close()
                        for i in range(3):
                            plt.subplot(3,1,i+1)
                            data = sacsL[-1][i].data
                            delta= sacsL[-1][i].stats['sac']['delta']
                            timeL = np.arange(len(data))*delta+sacsL[-1][i].stats['sac']['b']
                            plt.plot(timeL,data/data.std(),'b',linewidth=0.5)
                    if remove_resp and respDone==False and station['oRemove'] ==False:
                        print('remove_resp ',station)
                        
                        for channelIndex in range(len(strL)):
                            sac = sacsL[-1][channelIndex]
                            channelIndexO = defaultStrL.index(strL[channelIndex])
                            sensor = station.sensor[channelIndexO]
                            if self['time'] >= UTCDateTime(2009,7,1).timestamp and station['nameMode']=='CEA':
                                sensor = station.sensor[channelIndexO][:1]
                            if self['time'] < UTCDateTime(2009,7,1).timestamp and station['nameMode']=='CEA':
                                sensor = station.sensor[channelIndexO][1:]
                            das    = station.das[channelIndexO]
                            originStats={}
                            for key in station.defaultStats:
                                originStats[key] = sac.stats[key]
                            if station['net'] == 'hima':
                                sac.stats.update(station.defaultStats)
                            #print(sac.stats)
                            if station['net'] == 'YP' or station['nameMode'] == 'CEA':
                                sac.stats.update({'channel': station['compBase']+strL[channelIndex]})
                                sac.stats.update({'knetwk': station['net'],'network': station['net']})
                            #print(sac.stats,sensor[0][0][0],sensor[1][0][0])
                            if 'pre_filt' in para:
                                sac.remove_response(inventory=sensor,\
                                    output=para['output'],water_level=60,\
                                    pre_filt=para['pre_filt'])
                            else:
                                sac.remove_response(inventory=sensor,\
                                    output=para['output'],water_level=60)                           
                            sac.stats.update(station.defaultStats)
                            if station['nameMode'] != 'CEA':
                                sac.remove_response(inventory=das,\
                                    output="VEL",water_level=60)
                            sac.stats.update(originStats)
                        if isPlot:
                            for i in range(3):
                                plt.subplot(3,1,i+1)
                                data = sacsL[-1][i].data
                                delta= sacsL[-1][i].stats['sac']['delta']
                                timeL = np.arange(len(data))*delta+sacsL[-1][i].stats['sac']['b']
                                plt.plot(timeL,data/data.std(),'r',linewidth=0.5)
                    
                    for sac in sacsL[-1]:
                        sac.detrend()
                        sac.data -= sac.data.mean()

                    if para['toDisp']:
                        for sac in sacsL[-1]:
                            sac.integrate()
                            if np.random.rand()<0.01:
                                print('integrate to Disp')
                    
                    for sac in sacsL[-1]:
                        sac.detrend()
                        sac.data -= sac.data.mean()

                    if para['freq'][0] > 0  and station['doFilt']:
                        for sac in sacsL[-1]:
                            if para['filterName']=='bandpass':
                                if np.random.rand()<0.01:
                                    print('do do bandpass**************************')
                                sac.filter(para['filterName'],\
                                    freqmin=para['freq'][0], freqmax=para['freq'][1], \
                                    corners=para['corners'], zerophase=para['zerophase'])
                            elif para['filterName']=='highpass':
                                sac.filter(para['filterName'],\
                                    freq=para['freq'][0],  \
                                    corners=para['corners'], zerophase=para['zerophase'])
                            elif para['filterName']=='lowpass':
                                sac.filter(para['filterName'],\
                                    freq=para['freq'][0],  \
                                    corners=para['corners'], zerophase=para['zerophase'])
                    if isPlot:
                        plt.savefig(resSacNames[0]+'.jpg',dpi=200)
                    if isSave and remove_resp and station['oRemove'] ==False:
                        resSacNames = station.baseSacName(tmpDir,strL=strL,infoStr=respStr)
                        for i in range(3):
                            sacsL[-1][i].write(resSacNames[i],format='SAC')
                else:
                    sacsL.append(resSacNames)
        return sacsL
    def select(self,req):
        if 'time0' in req:
            if self['time']<req['time0']:
                return False
        if 'time1' in req:
            if self['time']>req['time1']:
                return False    
        if 'loc0' in req:
            dist = self.dist(req['loc0'])
            if 'maxDist' in req:
                if dist > req['maxDist']:
                    return False
            if 'minDist' in req:
                if dist < req['minDist']:
                    return False
        for record in self.records:
            if not record.select(req):
                self.records.pop(self.records.index(record))
        return True
    def __setitem__(self,key,value):
        super().__setitem__(key,value)
        if key == 'time' :
            self['strTime'] = UTCDateTime(self['time']).strftime('%Y:%m:%d %H:%M:%S.%f')
        if key =='HMS' and self['YMD']!='' and self['HMS']!='':
            self['time'] = UTCDateTime(self['YMD'] + ' ' + self['HMS'])



class QuakeL(list):
    def __init__(self,*argv,**kwargs):
        super().__init__()
        self.inQuake = {}
        self.inRecord= {}
        self.keys = ['#','*','q-','d-',' ',' ']
        if 'quakeKeysIn' in kwargs:
            self.inQuake['keysIn'] = kwargs['quakeKeysIn']
        if 'recordKeysIn' in kwargs:
            self.inRecord['keysIn'] = kwargs['recordKeysIn']
        if 'keys' in kwargs:
            self.keys = kwargs['keys']
        if 'quakeSplitKey' in kwargs:
            self.inQuake['splitKey'] = kwargs['quakeSplitKey']
        if 'recordSplitKey' in kwargs:
            self.inRecord['splitKey'] = kwargs['recordSplitKey']
        if len(argv)>0:
            if isinstance(argv[0],str):
                self.read(argv[0],**kwargs)
            elif isinstance(argv[0],list):
                for tmp in argv[0]:
                    self.append(tmp)
        if 'file' in kwargs:
            self.read(kwargs['file'],**kwargs)
    def __add__(self,self1):
        for quake in self1:
            self.append(quake)
        return self
    def __getitem__(self,index):
        quakesNew = super().__getitem__(index)
        if isinstance(index,slice):
            quakesNew = QuakeL(quakesNew)
            quakesNew.inQuake = self.inQuake
            quakesNew.inRecord = self.inRecord
            quakesNew.kyes = self.keys
        return quakesNew
    def find(self,quake0):
        for i in range(len(self)):
            dTime = np.abs(self[i]['time']-quake0['time'])
            dLa = np.abs(self[i]['la']-quake0['la'])
            dLo = np.abs(self[i]['lo']-quake0['lo'])
            if dTime<10 and dLa<0.5 and dLo<0.5:
                return i
        return -1
    def read(self,file,**kwargs):
        if 'keys' in kwargs:
            self.keys = kwargs['keys']
        if 'quakeSplitKey' in kwargs:
            self.inQuake['splitKey'] = kwargs['quakeSplitKey']
        if 'recordSplitKeys' in kwargs:
            self.inRecord['splitKey'] = kwargs['recordSplitKey']
        with open(file,'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line)<0:
                continue
            if line[0] == '^':
                self.keys = line[1:].split()
                if len(self.keys) >=6:
                    self.inQuake['splitKey']  = self.keys[4]
                    self.inRecord['splitKey'] = self.keys[5]
                continue
            if line[0] in self.keys[0]:
                self.inQuake['keysIn'] = line[1:]
                continue
            if line[0] in self.keys[1]:
                self.inRecord['keysIn'] = line[1:]
                continue
            if line[0] in self.keys[2]:
                self.inQuake['line'] = line
                self.append(Quake(**self.inQuake))
                continue
            if line[0] in self.keys[3]:
                continue
            #print(line[0],self.keys)
            self.inRecord['line'] = line
            self[-1].Append(Record(**self.inRecord))
    def write(self,file,**kwargs):
        if 'quakeSplitKey' in kwargs:
            self.inQuake['splitKey'] = kwargs['quakeSplitKey']
        if 'recordSplitKeys' in kwargs:
            self.inRecord['splitKey'] = kwargs['recordSplitKey']
        with open(file,'w+') as f:
            f.write('^')
            for key in self.keys:
                f.write(key+' ')
            f.write('\n')
            if 'quakeKeysIn' in kwargs:
                self.inQuake['keysIn'] = kwargs['quakeKeysIn']
            if 'recordKeysIn' in kwargs:
                self.inRecord['keysIn'] = kwargs['recordKeysIn']              
            if 'keysIn' in self.inQuake:
                tmp = self.inQuake['keysIn'].split()
                for quake in self:
                    self.keysIn = tmp
            if 'keysIn' in self.inRecord:
                tmp = self.inRecord['keysIn'].split()
                for quake in self:
                    for record in quake.records:
                        record.keysIn =  tmp
            quakeKeysIn = ''
            recordKeysIn= ''
            for quake in self:
                if quake.keyIn() != quakeKeysIn:
                    quakeKeysIn = quake.keyIn()
                    f.write('#%s\n'%quakeKeysIn)
                f.write('%s\n'%quake)
                for record in quake.records:
                    if record.keyIn() != recordKeysIn:
                        recordKeysIn = record.keyIn()
                        f.write('*%s\n'%recordKeysIn)
                    f.write('%s\n'%record)
    def select(self,req):
        for index in range(len(self)-1,-1,-1):
            quake = self[index]
            if not quake.select(req):
                self.pop(index)
            else:
                print('find ', quake)
        #quakes = self.copy()
        #self.clear()
        #for i in index:
        #    self.append(quakes[i])
    def cutSac(self, *argv,**kwargs):
        for quake in self:
            print(quake)
            quake.cutSac(*argv,**kwargs)
    def copy(self):
        quakes = QuakeL()
        for quake in self:
            quakes.append(quake.copy())
        quakes.keys         = self.keys.copy()
        quakes.inQuake      = self.inQuake.copy()
        quakes.inRecord     = self.inRecord.copy()
        return quakes
    def getSacFiles(self,*argv,**kwargs):
        sacsL = []
        for quake in self:
            sacsL+=quake.getSacFiles(*argv,**kwargs)
        return sacsL





def adjust(data,stloc=None,kzTime=None,tmpFile='test.sac',decMul=-1,\
    eloc=None,chn=None,sta=None,net=None,o=None):
    if decMul>1 :
        decMul = int(decMul)
        if decMul<16:
            data.decimate(int(decMul),no_filter=False)#True
        else:
            if decMul%2==0:
                data.decimate(2,no_filter=False)
                decMul/=2
            if decMul%2==0:
                data.decimate(2,no_filter=False)
                decMul/=2
            if decMul%5==0:
                data.decimate(5,no_filter=False)
                decMul/=5
            if decMul%5==0:
                data.decimate(5,no_filter=False)
                decMul/=5
            if decMul>1 :
                data.decimate(int(decMul),no_filter=False)
            if np.random.rand()<0.01:
                print(data)
    if data.stats['_format']!='SAC':
        data.write(tmpFile,format='SAC')
        data=obspy.read(tmpFile)[0]
        #print(data)
    if stloc!=None:
        data.stats['sac']['stla']=stloc[0]
        data.stats['sac']['stlo']=stloc[1]
        data.stats['sac']['stel']=stloc[2]
    if eloc!=None:
        data.stats['sac']['evla']=eloc[0]
        data.stats['sac']['evlo']=eloc[1]
        data.stats['sac']['evdp']=eloc[2]
        dis=DistAz(eloc[0],eloc[1],stloc[0],stloc[1])
        dist=dis.degreesToKilometers(dis.getDelta())
        data.stats['sac']['dist']=dist
        data.stats['sac']['az']=dis.getAz()
        data.stats['sac']['baz']=dis.getBaz()
        data.stats['sac']['gcarc']=dis.getDelta()
    if chn!=None:
        data.stats['sac']['kcmpnm']=chn
        data.stats['channel']=chn
    if sta!=None and net !=None:
        data.stats['sac']['kstnm']=sta
        data.stats['station']=sta
    if o!=None:
        data.stats['sac']['o']=o
    if kzTime!=None:
        kzTime=UTCDateTime(kzTime)
        data.stats['sac']['nzyear'] = int(kzTime.year)
        data.stats['sac']['nzjday'] = int(kzTime.julday)
        data.stats['sac']['nzhour'] = int(kzTime.hour)
        data.stats['sac']['nzmin']  = int(kzTime.minute)
        data.stats['sac']['nzsec']  = int(kzTime.second)
        data.stats['sac']['nzmsec'] = int(kzTime.microsecond/1000)
        data.stats['sac']['b']      = data.stats.starttime.timestamp-kzTime.timestamp
        data.stats['sac']['e']      = data.stats['sac']['b']+(data.data.size-1)*data.stats.delta
        #data.write(tmpFile)
        #data=obspy.read(tmpFile)[0]
        #print(data.stats['sac']['b'],data.stats['sac']['e'])
    return data



def mergeSacByName(sacFileNames, **kwargs):
    para ={\
    'delta0'    :0.02,\
    'freq'      :[-1, -1],\
    'filterName':'bandpass',\
    'corners'   :2,\
    'zerophase' :True,\
    'maxA'      :1e5,\
    }
    count       =0
    sacM        = None
    tmpSacL     =None
    para.update(kwargs)
    for sacFileName in sacFileNames:
        try:
            tmpSacs=obspy.read(sacFileName, debug_headers=True,dtype=np.float32)
            if para['freq'][0] > 0:
                tmpSacs.detrend('demean').detrend('linear').filter(para['filterName'],\
                        freqmin=para['freq'][0], freqmax=para['freq'][1], \
                        corners=para['corners'], zerophase=para['zerophase'])
            else:
                tmpSacs.detrend('demean').detrend('linear')
        except:
            print('wrong read sac')
            continue
        else:
            if tmpSacL==None:
                tmpSacL=tmpSacs
            else:
                tmpSacL+=tmpSacs
    if tmpSacL!=None and len(tmpSacL)>0:
        ID=tmpSacL[0].id
        for tmp in tmpSacL:
            try:
                tmp.id=ID
            except:
                pass
            else:
                pass
        try:
            sacM=tmpSacL.merge(fill_value=0)[0]
            std=sacM.std()
            if std>para['maxA']:
                print('#####too many noise std : %f#####'%std)
                sacM=None
            else:
                pass
        except:
            print('wrong merge')
            sacM=None
        else:
            pass
            
    return sacM

class Noises:
    def __init__(self,noisesL,mul=0.2):
        for noises in noisesL:
            for noise in noises:
                noise.data /= (noise.data.std()+1e-15)
        self.noisesL  = noisesL
        self.mul = mul
    def __call__(self,sacsL,channelL=[0,1,2]):
        for sacs in sacsL:
            for i in channelL:
                self.noiseSac(sacs[i],i)
    def noiseSac(self,sac,channelIndex=0):
        noise = random.choice(self.noisesL)[channelIndex]
        nSize = noise.data.size
        sSize = sac.data.size
        randI = np.random.rand()*nSize
        randL = (np.arange(sSize)+randI)%nSize
        sac.data+=(np.random.rand()**3)*noise.data[randL.astype(np.int)]\
        *self.mul*sac.data.std()


class PZ:
    def __init__(self,poles,zeros):
        self.poles=poles
        self.zeros=zeros
    def __call__(self,f):
        output = f.astype(np.complex)*0+1
        for pole in self.poles:
            output/=(f-pole)
        for zero in self.zeros:
            output*=(f-zero)
        return output

def sacFromO(sac):
    if sac.stats['sac']['b']>0:
        kzTime = sac.stats.starttime-sac.stats['sac']['b']
        d  = int((sac.stats['sac']['b']+1)/sac.stats['sac']['delta'])*\
            sac.stats['sac']['delta']
        dN = int(d/sac.stats['sac']['delta'])
        N  = sac.data.size
        data = np.zeros(dN+N)
        data[dN:] = sac.data
        sac.data = data
        sac.stats.starttime -=d
        sac = adjust(sac,kzTime=kzTime.timestamp)
    kzTime= sac.stats.starttime - sac.stats['sac']['b']
    endtime  = sac.stats.endtime
    sac=sac.slice(starttime=kzTime, \
        endtime=endtime, nearest_sample=True)
    sac=adjust(sac,kzTime=kzTime.timestamp)

    return sac

from obspy.taup import TauPyModel
from scipy import interpolate
iasp91 = TauPyModel(model="iasp91")

class taup:
    def __init__(self,quickFile='iasp91_time',phase_list=['P','p'], recal=False):
        self.dep = np.arange(0,1500,25)
        self.dist= np.arange(0.1,180,2)
        quickFileNew = quickFile
        for phase in phase_list:
                quickFileNew += phase
        if not os.path.exists(quickFileNew) or recal:
            self.genFile(quickFileNew,phase_list)
        self.M = np.loadtxt(quickFileNew)
        self.interpolate = interpolate.interp2d(self.dep,self.dist,self.M,\
            kind='linear',bounds_error=False,fill_value=1e-8)
    def genFile(self,quickFile='iasp91_timeP',phase_list=['P','p'] ):
        M = np.zeros([len(self.dist),len(self.dep)])
        time=0
        for i  in range(len(self.dep)):
            print(i)
            for j in range(len(self.dist)) :
                arrs=iasp91.get_travel_times(source_depth_in_km=self.dep[i],\
                    distance_in_degree=self.dist[j],\
                    phase_list=phase_list)
                if len(arrs)>0:
                    time = arrs[0].time
                else:
                    print(self.dep[i],self.dist[j],time)
                M[j,i] = time
                
        np.savetxt(quickFile,M)
    def __call__(self,dep,dist):
        return self.interpolate(dep,dist)