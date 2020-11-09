import os 
#os.system('echo %s|sudo -S %s' % ('Geodynamics', 'mkdir dirOld'))
with open('diskLst') as f:
	for line in f.readlines():
		uuid,point,DIR=line.split()
		if not os.path.exists(DIR):
			os.system('echo %s | sudo -S mkdir %s'\
			 % ('a314349798', DIR))
		os.system('echo %s | sudo -S mount /dev/disk/by-uuid/%s %s'\
		 % ('a314349798',uuid , DIR))
