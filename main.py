#!/usr/bin/python

import sys
import lib.config as conf
import lib.filereader as reader
import lib.preparedata as prepare
import lib.splitdataset as split
import models.regression.linearmodel as mdl
import models.classification.classificationmodel as cmdl

import warnings
warnings.filterwarnings("ignore")

def switchToLinear(config):
	inputFilePath = config['DEFAULT']['InputFilePath']
	inputFileDelimiter = config['DEFAULT']['InputFileDelimiter']
	columns = config['DEFAULT']['Columns']
	skipHeader = config['DEFAULT']['SkipHeader']
	targetColumn = config['REGRESSION']['TargetColumn']
	predictFilePath = config['REGRESSION']['PredictFilePath']

	data = reader.getDelimitedData(inputFilePath,inputFileDelimiter,columns,skipHeader)
	X,Y = prepare.getRegressionData(data,targetColumn)
	X_train, X_test, Y_train, Y_test = split.splitXYData(X,Y)
	model = mdl.evaluateLinearModels(X_train,Y_train,X_test,Y_test)
	model.fit(X,Y)
	predicted = model.predict(reader.getPredictData(predictFilePath))
	print(predicted)

def switchToClassification(config):
	inputFilePath = config['DEFAULT']['InputFilePath']
	inputFileDelimiter = config['DEFAULT']['InputFileDelimiter']
	columns = config['DEFAULT']['Columns']
	skipHeader = config['DEFAULT']['SkipHeader']
	targetColumn = config['CLASSIFICATION']['TargetColumn']
	predictFilePath = config['CLASSIFICATION']['PredictFilePath']
	
	data = reader.getDelimitedData(inputFilePath,inputFileDelimiter,columns,skipHeader)
	X,Y = prepare.getRegressionData(data,targetColumn)
	cmodel = cmdl.evaluateClassificationModels(X,Y)
	cmodel.fit(X,Y)
	predicted = cmodel.predict(reader.getPredictData(predictFilePath))
	print(predicted)

def main():
	if len(sys.argv)<2 :
		print("Invalid Number of Arguments")
		print("Usage: python main.py <config file path>")
		sys.exit(0)
	else :
		configFileName=sys.argv[1]
		config = conf.readConfigFile(configFileName)
		modelType = config['REGRESSION']['ModelType']
		if modelType == 'regression':
			switchToLinear(config)
		elif modelType == 'classification':
			switchToClassification(config)
		else:
			print("Invalid model type selected")
			sys.exit(0)
		
#This function is to call main function when executed
if __name__== "__main__":
  main()