
import numpy as np
import en
#import peach
import sklearn.feature_extraction.text as skltext

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

# Tokenizing the tweet
tokenizer = skltext.CountVectorizer(min_df=300)
neg_token = tokenizer.fit_transform(neg_tweet)
neg_token = neg_token.toarray()


# get the keyword names
analyze = tokenizer.build_analyzer()
neg_keyword = tokenizer.get_feature_names()
num_neg_keyword = len(neg_keyword)
print('Number of Negative Keywords = ' + str(num_neg_keyword) )
print neg_keyword


# remove keywords that are numbers
remove_key = []
for key in range(0, num_neg_keyword):

    if en.is_number(neg_keyword[key]):

        remove_key.append(key)
print ("These keywords are removed:")
print remove_key
print ('Number of keyword deleted ' + str(len(remove_key)))
neg_token = np.delete(neg_token, remove_key, 1)
neg_keyword = np.delete(neg_keyword, remove_key)
#print ("These keywords remained:")
#print neg_keyword


# count the number of keyword and tweets
neg_token_shape = neg_token.shape
num_neg_tweet = neg_token_shape[0]
num_neg_keyword = neg_token_shape[1]
print('Number of Negative Tweets = ' + str(num_neg_tweet) )
print('Number of Negative Keywords = ' + str(num_neg_keyword))
neg_token_occur = sum(neg_token)
keywords = dict(zip(neg_keyword, neg_token_occur))
print keywords

all_neg_tweet = '; '.join(neg_tweet[1:num_neg_tweet])
keywords = en.content.keywords(all_neg_tweet, top=50, nouns=False, singularize=True)
print keywords