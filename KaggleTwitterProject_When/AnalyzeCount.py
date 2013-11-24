__author__ = 'Ivan'
import pandas as p
import numpy as np
from collections import Counter
import nltk

paths = ['C:/MTE/train.csv', 'C:/MTE/test.csv'] #importing the data
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

Text = t['tweet']# Transform raw text documents to tfidf vectors. Learn a conversion law from documents to array data
test = np.load('test.npy') # Apply the transformation to the test data
y = np.array(t.ix[:,9:13])# ix divides the data in columns
#counts how many of verbs, nouns, etc are there
X=np.load('X.npy')

#algorithm used to solve
from sklearn import linear_model
clf =linear_model.Ridge(alpha=.1)
clf.fit(X,y)
test_prediction = clf.predict(test)

#print error
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/(X.shape[0]*4.0)))#

#save in the right format
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))#array: converts the list to array. Hstack: stack arrays horizontally

#saving format
col = '%i,' + '%f,'*3+ '%f'
np.savetxt('C:/MTE/prediction3.csv', prediction, col, delimiter=',')