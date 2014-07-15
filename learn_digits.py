
#Jon Hagg
#12/07/2014

#kaggle digit recognition
#random forest classifier

import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Load training data into numpy array
train   = np.loadtxt(open("../data/train.csv","rb"),delimiter=",",skiprows=1)
train_X = train[:,1:]
train_Y = train[:,0]


#Fit a classifier to training data
clf = RandomForestClassifier(n_estimators = 20, criterion='entropy')
clf.fit(train_X, train_Y)

#Load and predict classes for test data using our shiny new classifier, 
#write to csv file, formatted for kaggle
test=np.loadtxt(open("../data/test.csv","rb"), delimiter=",", skiprows=1)
pred = clf.predict(test)
pred=np.array([np.arange(1,len(pred)+1), map(int, pred)]).transpose()
np.savetxt('../results/output.csv', pred, fmt='%i', delimiter=',', header='ImageId,Label', comments='')


#Tests against benchmark classifier
#ONLY useful if we know accuracy of benchmark on test data

#bench=np.loadtxt(open("../data/rf_benchmark.csv","rb"), delimiter=",", skiprows=1)[:,1]
#score = clf.score(test, bench)
#print score

