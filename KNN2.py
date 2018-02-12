import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings, sys
from matplotlib import style
style.use('fivethirtyeight')
from collections import Counter
import pandas as pd
import random




# setup testing data
# data = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
# new_features = [5,7]
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in data[i]] for i in data]
# plt.scatter(new_features[0], new_features[1])
# plt.show()




def KNN(data, predict, k=3):
	
	# verify theres a valid k value
	if k <= len(data):
		print('k is <= total number of groups, and it needs to be less')
		sys.exit()

	# get euclidian distance between each the features of
	# each sample and the sample with new_features
	distances = []
	for group in data:
		for features in data[group]:

			# get euclidean distance
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])



	# get the k nearest neighbors to the sample with new_features
	votes = [i[1] for i in sorted(distances)[:k]]
	
	# get the 1 most common group of the k nearest neighbors
	final_vote = Counter(votes).most_common(1)[0][0]

	return final_vote

# # setup test
# result = KNN(data, new_features, k=3)
# print(result)



print('\nImporting data ...')
df = pd.read_csv('breast-cancer-wisconsin.txt') # import data from csv file

print('Formatting data ...')
df.replace('?', -99999, inplace=True) # remove missing data points
df.drop(['id'], 1, inplace=True) # remove useless data
full_data = df.astype(float).values.tolist() # convert everything to floats
random.shuffle(full_data) # shuffle all the samples

# separate training and test data
test_size = 0.20
train_set = {2:[], 4:[]}
test_set  = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for d in train_data:
	train_set[d[-1]].append(d[:-1])
for d in test_data:
	test_set[d[-1]].append(d[:-1])

# test network
print('Testing k nearest neighbor algorithm ...')
correct, total = 0, 0
for group in test_set:
	for features in test_set[group]:
		vote = KNN(train_set, features, k=5)
		# k=5 b/c sklearn uses a default value of 5
		if group==vote:
			correct += 1
		total += 1

print('Test Accuracy = %d/%d = %.3f%%' % (correct, total, (100*correct/total)))
