import sacTool
import tool
import names
from obspy import UTCDateTime

staInfos=sacTool.readStaInfos('staLstAll')
tool.getStaInArea(staInfos,'staLstSxh',[33.8,41,5,104,115])
staInfos=sacTool.readStaInfos('staLstSxh')
catalogFile='sxh'
with open(catalogFile) as f:
	for line in f.readlines():
		time=UTCDateTime(line[:23])
		print(time)
		sacTool.cutSacByDate(time,staInfos,\
			names.NMFileNameHour,outDir='SXH/',\
			decMul=-1,nameMode='event')

