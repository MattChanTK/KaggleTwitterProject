
import numpy as np
#import en
#import peach
#import sklearn.feature_extraction.text as skltext
import fea_extract
import import_data as ip
import evaluation as eval

train_data = ip.import_csv('../Data/train.csv')

num_train_sample = len(train_data)
print('Number of Training Data: ' + str(num_train_sample))

# only using a subset of the training set
#train_sample = train_sample[0:num_train_sample]
train_sample = train_data[0:int(num_train_sample-100)]
test_sample = train_data[int(num_train_sample-100): int(num_train_sample)]


# extract the tweet strings
train_tweet = ip.extract_tweet(train_sample)
test_tweet = ip.extract_tweet(test_sample)

# count the number of samples
train_num_sample = len(train_tweet)
print('Number of Training Tweet Subset: ' + str(train_num_sample))

# extract the sentiment counts
train_s = ip.extract_sentiment(train_sample)
test_s = ip.extract_sentiment(test_sample)
num_class = len(train_s[0])

keyword_list = []
num_sub_tweet = np.zeros(num_class,  dtype=int)

# compute keywords for each sentiment type
for class_type in range(0, num_class):

    sub_tweet = fea_extract.extract_tweet_subset(train_tweet, train_s, class_type)
    num_sub_tweet[class_type] = len(sub_tweet)
    print('\nNumber of S%d Tweets = ' % class_type + str(num_sub_tweet[class_type]))

    # extract keywords and their counts from the tweet
    token_counter = fea_extract.vectorize(min_occur=int(num_sub_tweet[class_type]/200))
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
keyword_list = fea_extract.scale_occur_counts(keyword_list, num_sub_tweet)

for i in range(0,num_class):
    print('Number of S%d Keywords = ' % i + str(len(keyword_list[i])))
    fea_extract.print_keyword(keyword_list[i], value_type='float')

# Merging and adding the keyword lists
print('\nMerging the keyword lists')
merged_keyword_list = fea_extract.merge_keyword_lists(keyword_list)
fea_extract.print_keyword(merged_keyword_list, value_type='list')

# Normalize the keywords to find the significant score for each sentiment class
print('\nCalculating significant scores')
merged_keyword_list = fea_extract.sig_score(merged_keyword_list)
fea_extract.print_keyword(merged_keyword_list, value_type='list')
print('Number of S%d Keywords = ' % i + str(len(merged_keyword_list)))

# Remove common keyword that has sizable percentage in all sentiment classes
print('\nRemoving Common Keywords')
merged_keyword_list = fea_extract.rm_common_keyword(merged_keyword_list, 0.1)
fea_extract.print_keyword(merged_keyword_list, value_type='list')
print('Number of S%d Keywords = ' % i + str(len(merged_keyword_list)))



# Generating similarity score as feature
print('\nGenerating similarity score as feature')
sim_scores = []
step = 0
for test_text in test_tweet:
    score = fea_extract.calc_similarity([test_text], merged_keyword_list, num_class)
    sim_scores.append(score)
    step += 1
    if step % 10 == 0:
        print step

for (i, tweet_content) in enumerate(test_tweet):
    print(tweet_content)
    for score in sim_scores[i]:
        print "%2.4f\t" % score,
    print ""

print eval.rmse(sim_scores, test_s)