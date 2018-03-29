#https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import sys
sys.path.append('../')
import models.costfunction.calculaterror as cost

def evaluateClassificationModels(X,Y):
	# prepare models
	models = []
	models.append(('Logistic Regression', LogisticRegression()))
	models.append(('Linear DiscriminantAnalysis', LinearDiscriminantAnalysis()))
	models.append(('KNeighbors Classifier', KNeighborsClassifier()))
	models.append(('DecisionTree Classifier', DecisionTreeClassifier()))
	models.append(('Gaussian NB', GaussianNB()))
	models.append(('SVM', SVC()))

	# evaluate each model in turn
	dictionary = {}
	for name, model in models:
		accuracy = cost.crossvalidation(model,X,Y)
		dictionary[name] = accuracy

	#select the model
	dict_sorted = [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get, reverse=True)]
	model_selected = str(dict_sorted[0][0])
	print("MODEL SELECTED: " ,model_selected)
	print("CROSS VALIDATION: ",dictionary[model_selected])

	#return model
	for name,model in models:
		if(name == model_selected):
			return model