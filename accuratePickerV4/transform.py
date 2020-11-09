#coding: UTF-8
import os
from obspy import read
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
dataDir='/NET/admin/YNSC/2/2015*/'
logFile='test.log10'
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


do()









