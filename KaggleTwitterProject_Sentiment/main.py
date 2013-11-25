
import numpy as np
#import en
#import peach
#import sklearn.feature_extraction.text as skltext
import fea_extract


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
train_s = ip.extract_sentiment(train_sample)
num_class = len(train_s[0])

keyword_list = []
num_sub_tweet = np.zeros(num_class,  dtype=int)

# computer keywords for each sentiment type
for class_type in range(0, num_class):

    sub_tweet = fea_extract.extract_tweet_subset(train_tweet, train_s, class_type)
    num_sub_tweet[class_type] = len(sub_tweet)
    print('\nNumber of S%d Tweets = ' % class_type + str(num_sub_tweet[class_type]))

    # extract keywords and their counts from the tweet
    token_counter = fea_extract.vectorize(min_occur=int(num_sub_tweet[class_type]/100))
    sub_count = fea_extract.count_token(token_counter, sub_tweet)

    # get the keyword names
    sub_keyword = fea_extract.get_keywords(token_counter)
    print sub_keyword

    # remove keywords that are bad
    sub_keyword, sub_count = fea_extract.filter_keywords(sub_keyword, sub_count)

    # count the number of keyword and tweets
    sub_count_shape = sub_count.shape
    num_sub_tweet[class_type] = sub_count_shape[0]
    num_sub_keyword = sub_count_shape[1]
    print('Number of S%d Tweets = ' % class_type + str(num_sub_tweet[class_type]) )
    print('Number of S%d Keywords = ' % class_type + str(num_sub_keyword))


    sub_keyword_list = fea_extract.keywords_list(sub_keyword, sub_count)
    fea_extract.print_keyword(sub_keyword_list)
    keyword_list.append(sub_keyword_list)

    print('Number of S%d Keywords = ' % class_type + str(len(sub_keyword_list)))

# scale the occurrence counts based on their respective the number of tweets
print('\nScaling the occurrence counts based on the number of tweets')
for i in range(0, num_class):
    num_keywords = len(keyword_list[i])

    for k, v in keyword_list[i].iteritems():
            keyword_list[i][k] = float(v)/float(num_sub_tweet[i])

    print('Number of S%d Keywords = ' % i + str(len(keyword_list[i])))
    fea_extract.print_keyword(keyword_list[i], value_type='float')

# Merging and adding the keyword lists
print('\nMerging the keyword lists')
merged_keyword_list = fea_extract.merge_keyword_lists(keyword_list)
fea_extract.print_keyword(merged_keyword_list, value_type='list')



