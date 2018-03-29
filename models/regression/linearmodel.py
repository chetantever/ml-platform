from sklearn import linear_model
import sys
sys.path.append('../')
import models.costfunction.calculaterror as cost

def evaluateLinearModels(X_train,Y_train,X_test,Y_test):
	# prepare models
	models = []
	models.append(('Linear Regression', linear_model.LinearRegression()))
	models.append(('Ridge', linear_model.Ridge (alpha = .5)))
	models.append(('RidgeCV', linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])))
	models.append(('Lasso', linear_model.Lasso(alpha = 0.1)))
	
	# evaluate each model 
	dictionary = {}
	for name, model in models:
		model.fit(X_train,Y_train)
		X_predict=model.predict(X_test)
		rmse = cost.mse(Y_test, X_predict)
		dictionary[name] = rmse
	
	#select the model which has least RMSE
	dict_sorted = [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get, reverse=True)]
	model_selected = str(dict_sorted[0][0])
	print("MODEL SELECTED:" ,model_selected)
	print("RMSE: ",dictionary[model_selected])

	#return model
	for name,model in models:
		if(name == model_selected):
			return model