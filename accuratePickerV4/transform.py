#coding: UTF-8
import os
from obspy import read,UTCDateTime
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Process, Manager,Pool
'''
'A01(鲁纳菁)'   'A06(云坪村)'   'B05(巧家营)'  'C05(竹寿镇)'   J03河玉村      'J08(围墙村)'
'A02（以则村)'   B01（紫牛村）   C01杉木村     'C06(稻谷乡)'   J04莲塘村
 A03（大桥乡）   B03（蒙古村）   C03迎春村      J01五星        J05（小寨村）
'A05(大海村)'    B04双河村      'C04(河边村)'   J02老厂        J06大水沟

'A01(鲁纳箐测震)'   B02大树脚测震  'C02(淌塘测震)'   'C06(稻谷测震)'   J04莲塘村          'J08(围墙村)'
'A04(小河塘)'       B04双河测震     C03迎春村         J01铅厂村       'J05(小寨村)'
'A05(大海)'         B06黑泥沟测震  'C04(河边村)'      J02老厂乡        J06大水沟
 A06（云坪村）     'C01(衫木)'     'C05(竹寿测震）'   J03河玉村        J07（新场村测震）

A04小河塘（测震）  B02大树脚（测震）  B06黑泥沟（测震）  J02老厂（测震）  J06大水勾（测震）
A05大海（测震）    B04河边（测震）    C03迎春（测震）    J03河玉（测震）  围墙村
A06云坪（测震）    B05巧家营（测震）  C04河边村（测震）  J04莲塘（测震）
B01紫牛（测震）    B05竹寿镇（测震）  C06稻谷（测震）    J05小寨（测震）


 A01          A06          B05（巧家营）  'C03(迎春村)'   J02           'J06(大水沟)'
 A02         'B01(紫牛)'  'B06(黑泥沟)'   'C04(河边村)'   J03            J07
'A03(大桥)'   B02         'C01(杉木)'      C05           'J04(莲塘村)'   J08（围墙村）
 A05          B04          C02             J01           'J05(小寨村)'

'DXA01(鲁纳箐测震)'  'DXA06(塘上村测震)'  'DXC01(杉木村测震)'  'DXC06(稻谷乡测震)'  'DXJ05(小寨村测震)'
'DXA02(以则村测震)'  'DXB01(紫牛村测震)'  'DXC02(淌塘村测震)'  'DXJ01(铅厂村测震)'  'DXJ06(大水沟测震)'
'DXA03(大桥乡测震)'  'DXB04(双河村测震)'  'DXC03(迎春村测震)'  'DXJ02(老厂乡测震)'  'DXJ07(新场村测震)'
'DXA04(小河塘测震)'  'DXB05(巧家营测震)'  'DXC04(河边村测震)'  'DXJ03(河玉村测震)'  'DXJ08(围墙村测震)'
'DXA05(大海村测震)'  'DXB06(黑泥沟测震)'  'DXC05(竹寿镇测震)'  'DXJ04(莲塘村测震)'

Dxa01  Dxa03  Dxb05  Dxj07  Dxy02  Dxy04  Dxy06  Dxy08  Dxy10  Dxy12  Dxy14  Dxy16  Dxy18
Dxa02  Dxa06  Dxj05  Dxy01  Dxy03  Dxy05  Dxy07  Dxy09  Dxy11  Dxy13  Dxy15  Dxy17  Dxy19

Dxa01/            Dxj07/            Dxy06（含强震）/  Dxy12/            Dxy18/
Dxa02/            Dxy01/            Dxy07/            Dxy13/            Dxy19/
Dxa03/            Dxy02/            Dxy08/            Dxy14/
Dxa06/            Dxy03/            Dxy09/            Dxy15/
Dxb05/            Dxy04/            Dxy10 （含强震）/ Dxy16/
Dxj05（含强震）/  Dxy05/            Dxy11/            Dxy17/
'''

from glob import glob
resDir = '/HOME/jiangyr/YNSC_SAC/'
dataDir='/NET/admin/YNSC/2/2014*/'
logFile='test.log11'
if not  os.path.exists(resDir):
	os.makedirs(resDir)
net = 'XU'
compL=['HHZ','HHN','HHE']
def do(tmpDir=dataDir,f=None):   
    print(tmpDir)
    if f==None:
        with open(logFile,'w+') as f:
            do(tmpDir,f)		
        return
    lastName = tmpDir.split('/')[-2][-1]
    if lastName == '0' or lastName=='9':       
        return
    for tmp in glob(tmpDir+'/*/'):
        do(tmp,f)
    for file in glob(tmpDir+'/*'):
        try:
            if os.path.isdir(file):
                continue
            if file[-1]=='/':
                continue
            if file[-1]=="f" or file[-1]=='d' or file[-1]=='0':
                if file[-1]=="f":
                    if file.split('.')[-2][-1]!='2':
                        continue
                if file[-1]=="d":
                    if file.split('.')[-2][-3:-1]!='HH':
                        continue
                sacs = read(file)
                for i in range(len(sacs)):
                    sac = sacs[i]
                    sacFormat= sac.stats['_format']
                    startTime = sac.stats['starttime']
                    station   = sac.stats['station']
                    nameHead = '%s/%s.%s/%s/%s/%s'%(resDir,net,station,startTime.strftime('%Y'),\
                            startTime.strftime('%Y%m%d'),startTime.strftime('%Y%m%d%H%M%S'))
                    if sacFormat=='MSEED' or sacFormat=='GCF':
                            channel = sac.stats['channel']
                    else:
                            channel = compL[i]
                    sacFile = nameHead+'.'+channel+'.sac'
                    sacDir = os.path.dirname(sacFile)
                    if not os.path.exists(sacDir):
                            os.makedirs(sacDir)
                    log = '%s %s %s %s %s %s'%(file,startTime,sacFile,net,station,channel)
                    if os.path.exists(sacFile):
                            log +='_repeat******'
                    print(log)
                    f.write(log+'%\n')
                    sac.write(sacFile,format='SAC')
        except:
            print(file+' wrong ^^^^^^^^^^^^^^^^^^^^\n')
            f.write(file+' wrong ^^^^^^^^^^^^^^^^^^^^\n')
        else:
            print(file+' done')


#do()
'/NET/admin/YNSC/2/2013年数据（201302--201402）/测震（201306-201309）/J08(围墙村)/20130721/20130721T2100_7091Z2.gcf \
2013-07-21T21:00:01.000000Z /HOME/jiangyr/YNSC_SAC//XU.7091/2013/20130721/20130721210001.HHZ.sac\
 XU 7091 HHZ%'
def getDirStaLst(file,staD):
    with open(file,'r') as f:
        for line in f.readlines():
            if 'wrong' in line:
                continue
            tmp = line.split()
            fileInfo = tmp[0]
            sta = tmp[-2]
            if sta not in staD:
                staD[sta]={}
            dirL = fileInfo.split('/')
            staInfo = ''
            for DIR in dirL[5:9]:
                if  '201' in DIR or '202' in DIR:
                    continue
                staInfo+='_'+DIR
            distFile = tmp[-4]
            time=int(UTCDateTime(distFile.split('/')[-1].split('.')[0][:8]).timestamp/86400)
            if len(staInfo)==0:
                print(fileInfo)
            staInfo += file
            if 'repeat' in line:
                staInfo += '_repeat'
            if staInfo not in staD[sta]:
                staD[sta][staInfo] = []
                print(sta,staInfo)
            if time not in staD[sta][staInfo]:
                staD[sta][staInfo].append(time)
'''
staD = {}
timeLim = [UTCDateTime('20120101').timestamp/86400,UTCDateTime('20201201').timestamp/86400]
for file in glob('test*log*'):
    print(file)
    getDirStaLst(file,staD)
plotDir = 'YNSC_data/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)
with open('staTransDirL','w+') as f:
    for sta in staD:
        f.write(sta+'\n')
        plt.figure(figsize=[20,4])
        ax =plt.axes()
        count =0
        yT = []
        yTL= []
        for DIR in staD[sta]:
            f.write(DIR+'\n')
            time= np.array(staD[sta][DIR])
            plt.plot(time,time*0+count,'.')
            plt.text(timeLim[0]+10,count,'%s %s'%(UTCDateTime(time.min()*86400).strftime('%Y%m%d%H%M%S'),UTCDateTime(time.max()*86400).strftime('%Y%m%d%H%M%S')))
            yT.append(count)
            yTL.append(DIR)
            count-=1
        ax.set_yticks(yT)
        ax.set_yticklabels(yTL)
        plt.xlim(timeLim)
        plt.title(sta)
        plt.savefig('%s/%s.jpg'%(plotDir,sta),dpi=300)
        plt.close()
'''

def getSacDis(l):
    staDir,staD,STAD=l
    STA= staDir.split('/')[-2]
    print(STA)
    count=0
    for yearDir in glob(staDir+'/20??/'):
        for dayDir in glob(yearDir+'/20??????/'):
            if count%30 ==0:
                print(dayDir)
            count+=1
            for file in glob(dayDir+'*sac'):
                head = read(file,headonly=True,format='SAC')[0]
                sta = head.stats['station']
                time=int(head.stats['starttime'].timestamp/86400)
                if sta not in staD:
                    staD[sta]={}
                if STA not in STAD:
                    STAD[STA]={}
                staInfo = STA    
                if staInfo not in staD[sta]:
                    staD[sta][staInfo] = []
                    print(sta,staInfo)
                if time not in staD[sta][staInfo]:
                    staD[sta][staInfo].append(time)
                if STA not in STAD:
                    STAD[STA]={}
                staInfo = sta
                if staInfo not in STAD[STA]:
                    STAD[STA][staInfo] = []
                    print('station',STA,staInfo)
                if time not in STAD[STA][staInfo]:
                    STAD[STA][staInfo].append(time)
def mvSac(sta0,sta1,staName0,bTime,eTime):
    bTime = UTCDateTime(bTime)
    eTime = UTCDateTime(eTime)
    count=0
    for yearDir in glob('/HOME/jiangyr/YNSCMOVE/'+sta0+'/*/'):
        for dayDir in glob(yearDir+'/20??????/'):
            if count%30 ==0:
                print(dayDir)
            count+=1
            for file in glob(dayDir+'*sac'):
                time = UTCDateTime(os.path.basename(file)[:8])
                if time<bTime or time>eTime:
                    continue
                head = read(file,headonly=True,format='SAC')[0]
                staName = head.stats['station']
                #print(staName,staName0)
                if staName==staName0:
                    newDir  = '/HOME/jiangyr/YNSCMOVE/'+sta1+yearDir[-6:]+dayDir[-10:]
                    if not os.path.exists(newDir):
                        os.makedirs(newDir)
                    cmd = 'mv %s %s%s'%(file,newDir,os.path.basename(file))
                    print(cmd)
                    os.system(cmd)
                

with Manager() as m:
    staD = dict()
    STAD =dict()
    arg =[]
    timeLim = [UTCDateTime('20120101').timestamp/86400,UTCDateTime('20201201').timestamp/86400]
    for staDir in glob('/HOME/jiangyr/YNSCMOVE/Dx*/'):
        print(staDir)
        arg.append([staDir,staD,STAD])
    '''
    #with Pool(10) as p:
    #    p.map(getSacDis,arg)
    '''
    for ARG in arg:
        getSacDis(ARG)
    plotDir = 'YNSC_data/'
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    for sta in staD:
        plt.figure(figsize=[20,4])
        ax =plt.axes()
        count =0
        yT = []
        yTL= []
        for DIR in staD[sta]:
            time= np.array(staD[sta][DIR])
            plt.plot(time,time*0+count,'.')
            plt.text(timeLim[0]+10,count,'%s %s'%(UTCDateTime(time.min()*86400).strftime('%Y%m%d%H%M%S'),UTCDateTime(time.max()*86400).strftime('%Y%m%d%H%M%S')))
            yT.append(count)
            yTL.append(DIR)
            count-=1
        ax.set_yticks(yT)
        ax.set_yticklabels(yTL)
        plt.xlim(timeLim)
        plt.title(sta+'move')
        plt.savefig('%s/%s_move.jpg'%(plotDir,sta),dpi=300)
        plt.close()
    for STA in STAD:
        plt.figure(figsize=[20,4])
        ax =plt.axes()
        count =0
        yT = []
        yTL= []
        for DIR in STAD[STA]:
            time= np.array(STAD[STA][DIR])
            plt.plot(time,time*0+count,'.')
            plt.text(timeLim[0]+10,count,'%s %s'%(UTCDateTime(time.min()*86400).strftime('%Y%m%d%H%M%S'),UTCDateTime(time.max()*86400).strftime('%Y%m%d%H%M%S')))
            yT.append(count)
            yTL.append(DIR)
            count-=1
        ax.set_yticks(yT)
        ax.set_yticklabels(yTL)
        plt.xlim(timeLim)
        plt.title('station'+STA+'move')
        plt.savefig('%s/station+%s_move.jpg'%(plotDir,STA),dpi=300)
        plt.close()

'''
mvSac('Dxj07','Dxc04','7038','20171113','20181007')
mvSac('Dxb05','Dxc05','7060','20150315','20150903')#**************
mvSac('Dxa05','Dxb01','7072','20150314','20171231')
mvSac('Dxj06','Dxy16','Dxj06','20181010','20190602')
'''



