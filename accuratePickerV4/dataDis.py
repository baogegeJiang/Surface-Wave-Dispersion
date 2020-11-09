import os 
import sacTool
import tool
import names
import numpy as np
from matplotlib import pyplot as plt
staInfos = sacTool.readStaInfos('stations/staLstAll')
staFileLst = names.staFileLst
resDir = 'dataDis/'
if not os.path.exists(resDir):
    os.mkdir(resDir)
laLim=[30,45]
loLim=[92,118]
for i in range(17,20):
    for j in range(0,367,10):
        la = []
        lo = []
        dayKey = '%d.%03d_BH'%(i,j)
        for staInfo in staInfos:
            if staInfo['sta'] in staFileLst:
                if dayKey in staFileLst[staInfo['sta']]:
                    la.append(staInfo['la'])
                    lo.append(staInfo['lo'])
        if len(la)>0:
            plt.close()
            plt.plot(np.array(lo),np.array(la),'.')
            plt.title(dayKey)
            plt.xlim(loLim)
            plt.ylim(laLim)
            plt.savefig('%s/%d.%03d.jpg'%(resDir,i,j),dpi=200)
            plt.close()



