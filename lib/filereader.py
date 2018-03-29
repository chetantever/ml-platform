import pandas as pd

def getDelimitedData(filename,delimter,columns,skipHeader):
	if(skipHeader.upper() == "TRUE"):
		data = pd.read_csv(filename,sep=delimter,names=columns.split(','),skiprows=[0])
	else:
		data = pd.read_csv(filename,sep=delimter,names=columns.split(','))
	data=data.dropna()
	return data;

def getPredictData(filename):
	data = pd.read_csv(filename,header = None)
	data=data.dropna()
	return data