# k-Nearest_Neighbor

This code uses a k-Nearest_Neighbor (KNN) algorithm for Tumor Classification from breast cancer data, using both the sklearn library (KNN1.py) and by making it from scratch (KNN2.py).

The data is aquired from the University of California Irvine and can be found here:
http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

The KNN algorithm classifies new data points relative to the classifications of its training data by counting the classifications of the k data points closest to the new data point according to euclidean space (sklearn's default k value is 5), and classifying the new data point as whichever classification occured most often in its neighbors. For the breast-cancer-wisconsin data this KNN algorithm yeilded about a 96% accuracy for classifying tumors as benign or malignant for both the one using sklearn, and the one written from scratch, however the one written from scratch runs slower.

Another interesting fact about the KNN is that it can perform classifications on non-linear data. However, since KNNs have to search for the k nearest neighbors for each test sample by iterating through all the training data (sklearn has a few ways around this) it generally becomes very slow for EXTREMELY large data sets. 


# Sources
https://www.youtube.com/watch?v=1i0zu9jHN6U
