from glob import glob
import tool
import sacTool
import scpTool
import os
staInfos=sacTool.readStaInfos('stations/staLstAll')
staFileLst=tool.loadFileLst(staInfos,'include/fileLst')
compL={'BHE':'3','BHN':'2','BHZ':'1'}
def NMFileName(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
            for file in glob(sacFileNamesStr):
                sacFileNames.append(file)
            return sacFileNames
    comp=compL[comp]
    if station in staFileLst:
        if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
            staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
            fileP=staDir+'/??/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            sacFileNames=sacFileNames+glob(fileP)
    return sacFileNames

def NMFileNameScp(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
    comp=compL[comp]
    if station in staFileLst:
        if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
            staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
            fileP=staDir+'/??/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            sacFileNamesStr=fileP
    tmpDir='tmp/'+comp+'/'
    if os.path.exists(tmpDir):
        try:
            os.remove(tmpDir+'*[m,E,N,Z,C,c]')
        except:
            pass
        else:
            pass
    else:
        os.makedirs(tmpDir)
    return scpTool.get(sacFileNamesStr,tmpDir)

def NMFileNameHour(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
            for file in glob(sacFileNamesStr):
                sacFileNames.append(file)
            return sacFileNames
    comp0=comp
    sacFileNames = list()
    comp=compL[comp]
    if station in staFileLst:
        if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
            staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
            fileP=staDir+'/'+YmdHMSJ['H']+'/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            sacFileNames=sacFileNames+glob(fileP)
            Hour=(int(YmdHMSJ['H'])+1)%24
            H='%02d'%Hour
            fileP=staDir+'/'+H+'/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            sacFileNames=sacFileNames+glob(fileP)
    if len(sacFileNames)==0:
        sacDir='/media/jiangyr/Hima_Bak/hima31/'
        fileP=sacDir+YmdHMSJ['Y']+YmdHMSJ['m']+YmdHMSJ['d']+\
        '.'+YmdHMSJ['J']+'*/*.'+station+'*.'+comp0
        sacFileNames=sacFileNames+glob(fileP)
    if len(sacFileNames)==0:
        sacDir='/media/jiangyr/shanxidata2/hima31_2/'
        fileP=sacDir+YmdHMSJ['Y']+YmdHMSJ['m']+YmdHMSJ['d']+\
        '.'+YmdHMSJ['J']+'*/*.'+station+'*.'+comp0
        sacFileNames=sacFileNames+glob(fileP)
    return sacFileNames

def FileName(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    c=comp[-1]
    if c=='Z':
        c='U'
    sacFileNames.append('wlx/data/'+net+'.'+station+'.'+c+'.SAC')
    #print(sacFileNames)
    return sacFileNames
def NMFileNameBe(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
            for file in glob(sacFileNamesStr):
                sacFileNames.append(file)
            return sacFileNames
    comp=compL[comp]
    if station in staFileLst:
        if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
            staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
            fileP=staDir+'/??/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            for file in glob(fileP):
                if float(os.path.dirname(file).split('/')[-1])<12:
                    sacFileNames=sacFileNames+[file]
    return sacFileNames
def NMFileNameAf(net, station, comp, YmdHMSJ):
    sacFileNames = list()
    Y = YmdHMSJ
    if net == 'GS' or net=='NM':
            dir = '/media/jiangyr/shanxidata2/nmSacData/'
            staDir = dir+net+'.'+station+'/'
            YmDir = staDir+Y['Y']+Y['m']+'/'
            sacFileNamesStr = YmDir+net+'.'+station+'.'+Y['Y']+Y['m']+Y['d']+'.*'+comp+'*SAC'
            for file in glob(sacFileNamesStr):
                sacFileNames.append(file)
            return sacFileNames
    comp=compL[comp]
    if station in staFileLst:
        if YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH' in staFileLst[station]:
            staDir=staFileLst[station][YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'_BH']
            fileP=staDir+'/??/'+YmdHMSJ['Y'][2:4]+'.'+YmdHMSJ['j']+'*'+comp+'.m'
            for file in glob(fileP):
                if float(os.path.dirname(file).split('/')[-1])>=12:
                    sacFileNames=sacFileNames+[file]
    return sacFileNames

def SCFileName(net,station,comp,Y):
    #XX.QCH.2008209000000.BHZ
    fileName='/home/jiangyr/WC_mon78/%s.%s.%s%s000000.%s'%(net,station,Y['Y'],\
            Y['j'],comp)
    print(fileName)
    return [fileName]

def NEName(net,station,comp,Y):
    #/media/commonMount/data2/NECESSARRAY_SAC/NEdata_09-10/NE67/2009/R304/NE67.2009.304.00.00.00.BHZ.sac
    dataDir='/media/commonMount/data2/NECESSARRAY_SAC/'
    #print('%s/NEdata*/%s/%s/R%s/%s.%s.%s.*.%s.sac'%(dataDir,\
    #        station,Y['Y'],Y['j'],station,Y['Y'],Y['j'],comp))
    return glob('%s/NEdata*/%s/%s/R%s/%s.%s.%s.*.%s.sac'%(dataDir,\
            station,Y['Y'],Y['j'],station,Y['Y'],Y['j'],comp))

def CEAName(net,station,comp,Y):
    print( '/media/commonMount/CEA?/net_??/%s/%s/%s/%s.D/%s.%s.00.%s.D.%s.%s'\
                        %(Y['Y'],net,station,comp,net,station,comp,Y['Y'],Y['j']))
    return glob( '/media/commonMount/CEA?/net_??/%s/%s/%s/%s.D/%s.%s.00.%s.D.%s.%s'\
            %(Y['Y'],net,station,comp,net,station,comp,Y['Y'],Y['j']))
