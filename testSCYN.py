import obspy 
from glob import glob
import os
import seism

stations = seism.StationL('stations/SCYN')
#/media/commonMount/CEA0/net_yn/2014/YN/BAS/BHE.D
stationD = {}
fileL = glob('/media/commonMount/CEA?/net_??/20??/??/*/??E.D/*.20??.?01')
for fileName in fileL:
    baseName = os.path.basename(fileName)
    netSta  = baseName.split('.')
    netStaStr = netSta[0] + '.' + netSta[1]
    if netStaStr in stationD:
        continue
    else:
        stationD[netStaStr] = fileName
    compDir = fileName.split('/')[-2]
    baseComp= compDir[:2]
    station = stations.Find(netStaStr)
    station['compBase'] = baseComp
    station['sensorName'] = 'CEA_resp/RESP.%s.%s.00.BHZ'%\
    (station['net'],station['sta'])
    station['dasName'] = '130S'

stations.write('stations/SCYN_withComp',\
    'net sta compBase la lo erroLo erroLa dep erroDep sensorName dasName')
stations.write('stations/SCYN_withComp_ac',\
    'net sta compBase lo la erroLo erroLa dep erroDep')