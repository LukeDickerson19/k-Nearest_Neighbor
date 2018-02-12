import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd


print('\nImporting data ...')
df = pd.read_csv('breast-cancer-wisconsin.txt') # import data from csv file

print('Formatting data ...')
df.replace('?', -99999, inplace=True) # remove missing data points
df.drop(['id'], 1, inplace=True) # remove useless data
X = np.array(df.drop(['class'], 1)) # features
Y = np.array(df['class']) # labels 

# use 20% of data for testing 
X_train, X_test, Y_train, Y_test = \
cross_validation.train_test_split(X, Y, test_size=0.20)

print('Creating k-Nearest Neighbors Classifier ...')
classifier = neighbors.KNeighborsClassifier() # create k neighbors classifier

print('Training ...')
classifier.fit(X_train, Y_train) # train k neighbors classifier

print('Testing ...')
accuracy = classifier.score(X_test, Y_test) # test k neighbors classifier
print('Test Accuracy = %.3f %%' % (100 * accuracy)) # output the classifiers test accuracy

print('Making a prediction ...')
example_features = np.array([[4,2,1,1,1,2,3,2,1]])
prediction = classifier.predict(example_features)
print('Prediction for  %s is %s' % (example_features[0], prediction))


