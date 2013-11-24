#__author__ = 'Ivan'
#introduction to sklearn and python
import spell
import pandas as p
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import nltk

paths = ['C:/MTE/train.csv', 'C:/MTE/test.csv'] #importing the data
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])
#print t # display the data from the

cleanTweetsTrain=t['tweet']
for Word in cleanTweetsTrain:
    Word.translate(None,'!`~$%^&*()-_=+[]{}":;?/')#still to clean # and @
    Word.replace('@mention','')#Still need to clean #...

cleanTweetsTest=t2['tweet']
for Word in cleanTweetsTrain:
    Word.translate(None,'!`~$%^&*()-_=+[]{}":;?/')#still to clean # and @
    Word.replace('@mention', '') #still need to clean #...

tfidf = TfidfVectorizer(max_features=15000, strip_accents='unicode', analyzer='word')#Max features-maximum number of words obtained,
#Strip_accents=remove accents on all the words, analyzer-generate features based on words.
tfidf.fit(cleanTweetsTrain) #Learn conversion law. Apply Tf-Idf on the tweet column
X = tfidf.transform(cleanTweetsTrain)# Transform raw text documents to tfidf vectors. Learn a conversion law from documents to array data
test = tfidf.transform(cleanTweetsTest) # Apply the transformation to the test data
y = np.array(t.ix[:, 4:])# ix divides the data in columns

##new data
#test2= sparse.csr_matrix(np.load('test.npy')) # Apply the transformation to the test data
#test=sparse.hstack([test_initial, test2])
#X2 = sparse.csr_matrix(np.load('X.npy'))
#X = sparse.hstack([Y, X2])
#algorithm used to solve
from sklearn import linear_model
clf =linear_model.Ridge(alpha=.2)
clf.fit(X,y)
test_prediction = clf.predict(test)

#cleaning negative values
test_prediction = test_prediction.clip(0.0,1.0)

#print error
print 'Train error: {0}'.format(np.sqrt(np.sum(np.array(np.array(clf.predict(X))-y)**2)/(X.shape[0]*24.0)))#

#save in the right format
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction]))#array: converts the list to array. Hstack: stack arrays horizontally

#matrix: creates a matrix (increase from 1 to more dimensions. T:transpose
col = '%i,' + '%f,'*23+ '%f'
np.savetxt('C:/MTE/prediction6.csv', prediction, col, delimiter=',')

#import re, collections
#
#import nodebox
#import en
#import nltk
# separate a sentence in separate words
#sentence = """At eight o'clock on Thursday morning
#Arthur didn't feel very good."""
#tokens = nltk.word_tokenize(sentence)
#print tokens

#print en.is_basic_emotion("happy")

#tokens = nltk.word_tokenize('ivan is a great person forever and he feels happy')
#text = nltk.Text(tokens)
#tags = nltk.pos_tag(text)
#tagged = [('the', 'DT'), ('dog', 'NN'), ('sees', 'VB'), ('the', 'DT'), ('cat', 'NN')]
#from collections import Counter
#counts = Counter(tag for word,tag in tagged)
#print tokens


#sentence = """At eight o'clock on Thursday morning
#Arthur didn't feel very good."""
#tokens = nltk.word_tokenize(sentence)
#print tokens
#for l in range(7,10):
#    y=en.is_verb(tokens[l])
#    print y
#text = nltk.word_tokenize("And now for something completely different")
#nltk.pos_tag(text)

#counts how many of verbs, nouns, etc are there
#text = nltk.word_tokenize("hi, how are you, what have you been up to?")
#a=nltk.pos_tag(text)
#print a
#from collections import Counter
#counts = Counter(tag for word,tag in a)
#print counts
#print counts['VBD'] #print the number of VBD values
#print counts['VBZ'] #print the number of VB values

# prints a value if the the second part of the tuple is VBD
#b=list()##
#for i,j in a:
#    if j=='VBD':
#        print i
#        a.append(i)
#print b##

