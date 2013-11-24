__author__ = 'Ivan'
import pandas as p
import numpy as np
from collections import Counter
import nltk

paths = ['C:/MTE/train.csv', 'C:/MTE/test.csv'] #importing the data
t = p.read_csv(paths[0])
t2 = p.read_csv(paths[1])

Text = t['tweet']# Transform raw text documents to tfidf vectors. Learn a conversion law from documents to array data
test = t2['tweet'] # Apply the transformation to the test data
y = np.array(t.ix[:, 9:13])# ix divides the data in columns
#counts how many of verbs, nouns, etc are there
sz=len(test)
X=np.zeros(shape=(sz,6))

for index, t in enumerate(test):
    text = nltk.word_tokenize(t)
    a=nltk.pos_tag(text)
    counts = Counter(tag for word,tag in a)
    X[index, 0]=counts['VB']#Base form
    X[index, 1]=counts['VBD']#Past tense
    X[index, 2]=counts['VBG']#Gerund or present participle
    X[index, 3]=counts['VBN']#Past participle
    X[index, 4]=counts['VBP']+counts['VBZ']#Present non-third person+third person
    X[index, 5]=counts['MD']#Modal
    if index%1000==0:
        print index
np.save('test',X)
