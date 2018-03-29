from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection

def mse(Y_test,X_predict):
	mse=mean_squared_error(Y_test, X_predict)
	rmse=sqrt(mse)
	return rmse

#Cross Validation - attempts to avoid overfitting
def crossvalidation(model,X,Y):
	seed = 7
	scoring = 'accuracy'
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	return cv_results.mean()*100
