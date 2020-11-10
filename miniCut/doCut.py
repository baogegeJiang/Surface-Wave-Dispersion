import seism
stations = seism.StationList('staL')
quakes=seism.QuakeL('quakeL')
para={\
'delta0' :1,
'freq'   :[-1,-1],
'corners':4,
'maxA':1e18,
}
for quake in quakes:
	quake.keysName=['strTime','la','lo']

	
quakes.cutSac(stations,bTime=-1500,eTime =12300,\
    para=para,byRecord=False,isSkip=True,resDir='ba/')

quakes[0].name('_')