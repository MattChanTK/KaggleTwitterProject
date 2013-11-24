
import numpy as np
#import en
#import peach
#import sklearn.feature_extraction.text as skltext
import fea_extract
from pprint import pprint

import import_data as ip

train_sample = ip.import_csv('../Data/train.csv')

num_train_sample = len(train_sample)
print('Number of Training Data: ' + str(num_train_sample))

# only using a subset of the training set
train_sample = train_sample[0:num_train_sample]

# extract the tweet strings
train_tweet = ip.extract_tweet(train_sample)
# count the number of samples
train_num_sample = len(train_tweet)
print('Number of Training Tweet Subset: ' + str(train_num_sample))

# extract the sentiment counts
train_s = ip.extract_sentiment(train_sample);

#find the class with max membership
train_label = []
for i in train_s:
    train_label.append(np.argmax(i))


# extract only the negative tweet
neg_tweet = []
for i in range(0, train_num_sample):
    if train_label[i] == 1:
        neg_tweet.append(train_tweet[i])
num_neg_tweet = len(neg_tweet)
print('Number of Negative Tweets = ' + str(num_neg_tweet) )

# extract keywords and their counts from the tweet
token_counter = fea_extract.vectorize(min_occur=30)
neg_count = fea_extract.count_token(token_counter, neg_tweet)

# get the keyword names
neg_keyword = fea_extract.get_keywords(token_counter)
print neg_keyword

# remove keywords that are bad
neg_keyword, neg_count = fea_extract.filter_keywords(neg_keyword, neg_count)


# count the number of keyword and tweets
neg_count_shape = neg_count.shape
num_neg_tweet = neg_count_shape[0]
num_neg_keyword = neg_count_shape[1]
print('Number of Negative Tweets = ' + str(num_neg_tweet) )
print('Number of Negative Keywords = ' + str(num_neg_keyword))



neg_keyword_list = fea_extract.keywords_list(neg_keyword, neg_count)
fea_extract.print_keyword(neg_keyword_list)


#all_neg_tweet = '; '.join(neg_tweet[1:num_neg_tweet])
#keywords = en.content.keywords(all_neg_tweet, top=50, nouns=False, singularize=True)
#print keywords