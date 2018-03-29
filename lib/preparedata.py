import pandas as pd

def getRegressionData(data,targetColumn):
	X = data.drop(targetColumn,axis = 1)
	Y = data[targetColumn]
	return X,Y