import csv
import math
import numpy as np
import en
import peach
import sklearn.feature_extraction.text as skltext

train_csv = open('../Data/train.csv', 'rb')
train_data = csv.reader(train_csv, delimiter=',', quotechar='"')

train_sample = tuple(train_data)
# only using a subset of the training set
# train_sample = train_sample[1:1000]


def column(matrix, i):
    return [row[i] for row in matrix]


# extract the tweet strings
train_tweet = column(train_sample, 1)
# remove the heading
train_tweet.pop(0)
# count the number of samples
train_num_sample = len(train_tweet)

# extract the sentiment counts
train_s = []
for row in train_sample:
    train_s.append(row[4:9])
# remove the heading
train_s.pop(0)
#count the number of class
num_class = len(train_s[0])
#convert the labels to float
for i in range(0, train_num_sample):
    for j in range(0, num_class):
        train_s[i][j] = float(train_s[i][j])

#find the class with max membership
train_label = []
for i in train_s:
    train_label.append(np.argmax(i))


# extract only the negative tweet
neg_tweet = []
for i in range(0, train_num_sample):
    if train_label[i] == 1:
        neg_tweet.append(train_tweet[i])
#num_neg_tweet = len(neg_tweet)

# Tokenizing the tweet
tokenizer = skltext.CountVectorizer(min_df=2)
neg_token = tokenizer.fit_transform(neg_tweet)
neg_token = neg_token.toarray()
# count the number of keyword and tweets
neg_token_shape = neg_token.shape
num_neg_keyword = neg_token_shape[0]
num_neg_tweet = neg_token_shape[1]

# get the keyword names
analyze = tokenizer.build_analyzer()
neg_keyword = tokenizer.get_feature_names()
#num_neg_keyword = len(neg_keyword)
print neg_keyword


# remove keywords that are numbers
remove_key = []
for key in range(0, num_neg_keyword):
    if en.is_number(neg_keyword[key]):
        print(neg_keyword[key] + ' removed')
        remove_key.append(key)

neg_token = np.delete(neg_token, remove_key, 1)
neg_keyword = np.delete(neg_keyword, remove_key)
print neg_keyword
print neg_token
