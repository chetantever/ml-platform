from sklearn.cross_validation import train_test_split

testSize = 0.3
randomState = 100

def splitXYData(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = testSize, random_state = randomState)
	return X_train, X_test, y_train, y_test