__author__ = 'Ivan'
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as p

paths = ['C:/MTE/train.csv', 'C:/MTE/test.csv'] #importing the data
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

tfidf = TfidfVectorizer(max_features=15000, strip_accents='unicode', analyzer='word')#Max features-maximum number of words obtained,
#Strip_accents=remove accents on all the words, analyzer-generate features based on words.
tfidf.fit(t['tweet']) #Learn conversion law. Apply Tf-Idf on the tweet column
X = tfidf.transform(t['tweet'])# Transform raw text documents to tfidf vectors. Learn a conversion law from documents to array data
test = tfidf.transform(t2['tweet']) # Apply the transformation to the test data


#Sentiment
y = np.array(t.ix[:, 4:9])# ix divides the data in columns

#algorithm used to solve
from sklearn import linear_model
clf =linear_model.Ridge(alpha=.1)
clf.fit(X,y)
test_prediction = clf.predict(test)

#print error
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/(X.shape[0]*5.0)))#

#save in the right format
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))#array: converts the list to array. Hstack: stack arrays horizontally

#matrix: creates a matrix (increase from 1 to more dimensions. T:transpose
col = '%i,' + '%f,'*4+ '%f'
np.savetxt('C:/MTE/sentiment.csv', prediction, col, delimiter=',')




#when
y = np.array(t.ix[:, 9:13])# ix divides the data in columns

#algorithm used to solve
from sklearn import linear_model
clf =linear_model.Ridge(alpha=.1)
clf.fit(X,y)
test_prediction = clf.predict(test)

#print error
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/(X.shape[0]*4.0)))#

#save in the right format
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))#array: converts the list to array. Hstack: stack arrays horizontally

#matrix: creates a matrix (increase from 1 to more dimensions. T:transpose
col = '%i,' + '%f,'*3+ '%f'
np.savetxt('C:/MTE/when.csv', prediction, col, delimiter=',')



#Type
y = np.array(t.ix[:, 13:28])# ix divides the data in columns

#algorithm used to solve
from sklearn import linear_model
clf =linear_model.Ridge(alpha=.1)
clf.fit(X,y)
test_prediction = clf.predict(test)

#print error
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/(X.shape[0]*15.0)))#

#save in the right format
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))#array: converts the list to array. Hstack: stack arrays horizontally

#matrix: creates a matrix (increase from 1 to more dimensions. T:transpose
col = '%i,' + '%f,'*14+ '%f'
np.savetxt('C:/MTE/type.csv', prediction, col, delimiter=',')