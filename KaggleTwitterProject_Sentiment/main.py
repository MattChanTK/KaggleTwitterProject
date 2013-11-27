
import numpy as np
import cPickle as pickle
#from scipy import sparse
#from sklearn.feature_extraction.text import TfidfVectorizer
#import en
#import peach
#import sklearn.feature_extraction.text as skltext
import fea_extract
import import_data as ip
import evaluation as eval
import classification as clf

# loading data from files
train_data = ip.import_csv('../Data/train.csv')


# loading the pre computed data
classes = ['s', 'w', 'c']
saved_keywords_list = []
saved_train_fea = []
saved_test_fea = []

for c in classes:
    try:
        with open('keywords_list_'+c+'.p', 'rb') as fp:
            saved_keywords_list.append(pickle.load(fp))
        #saved_keywords_list.append(np.load('keywords_list_'+c+'.npy'))
        saved_train_fea.append(np.load('train_fea_'+c+'.npy'))
        saved_test_fea.append(np.load('test_fea_'+c+'.npy'))

    except IOError:
        saved_keywords_list.append([])
        saved_train_fea.append([])
        saved_test_fea.append([])

num_train_sample = len(train_data)
print('Number of Training Data: ' + str(num_train_sample))

header = ip.extract_header(train_data)

'''
# only using a subset of the training set
#train_sample = train_sample[0:num_train_sample]
train_sample = train_data[0:int(num_train_sample)]
test_sample = train_data[int(num_train_sample-2000): int(num_train_sample)]
'''
num_data = 10000
train_sample = train_data[0:int(num_data)]
test_sample = train_data[int(num_data): int(num_data+100)]


# extract the tweet strings
train_tweet = ip.extract_tweet(train_sample)
test_tweet = ip.extract_tweet(test_sample)

#extract the tweet id
train_id = map(int, ip.extract_tweet_id(train_sample))
test_id = map(int, ip.extract_tweet_id(test_sample))

# count the number of samples
train_num_sample = len(train_tweet)
print('Number of Training Tweet Subset: ' + str(train_num_sample))


all_train_fea = []
all_test_fea = []

for c in range(0, len(classes)):
    # extract the sentiment counts
    train_mem = ip.extract_membership(train_sample, class_type=classes[c])
    num_class = len(train_mem[0])

    keyword_list = []
    num_sub_tweet = np.zeros(num_class,  dtype=int)

    if saved_keywords_list[c]:
        merged_keyword_list = saved_keywords_list[c]
    else:
        # compute keywords for each sentiment type
        for class_type in range(0, num_class):

            sub_tweet = fea_extract.extract_tweet_subset(train_tweet, train_mem, class_type)
            num_sub_tweet[class_type] = len(sub_tweet)
            print('\nNumber of %s%d Tweets = ' % (classes[c], class_type) + str(num_sub_tweet[class_type]))

            # extract keywords and their counts from the tweet
            token_counter = fea_extract.vectorize(min_occur=max(int(num_sub_tweet[class_type]/(1000/num_class)), 2))
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
            print('Number of %s%d Tweets = ' % (classes[c], class_type) + str(num_sub_tweet[class_type]))
            print('Number of %s%d Keywords = ' % (classes[c], class_type) + str(num_sub_keyword))


            sub_keyword_list = fea_extract.keywords_list(sub_keyword, sub_count)
            fea_extract.print_keyword(sub_keyword_list)
            keyword_list.append(sub_keyword_list)

            print('Number of %s%d Keywords = ' % (classes[c], class_type) + str(len(sub_keyword_list)))

        # scale the occurrence counts based on their respective the number of tweets
        print('\nScaling the occurrence counts based on the number of tweets')
        keyword_list = fea_extract.scale_occur_counts(keyword_list, num_sub_tweet)

        for i in range(0,num_class):
            print('Number of %s%d Keywords = ' % (classes[c], i) + str(len(keyword_list[i])))
            fea_extract.print_keyword(keyword_list[i], value_type='float')

        # Merging and adding the keyword lists
        print('\nMerging the keyword lists')
        merged_keyword_list = fea_extract.merge_keyword_lists(keyword_list)
        fea_extract.print_keyword(merged_keyword_list, value_type='list')

        # Normalize the keywords to find the significant score for each sentiment class
        print('\nCalculating significant scores')
        merged_keyword_list = fea_extract.sig_score(merged_keyword_list)
        fea_extract.print_keyword(merged_keyword_list, value_type='list')
        print('Number of %s Keywords = ' % classes[c] + str(len(merged_keyword_list)))

        # Remove common keyword that has sizable percentage in all sentiment classes
        print('\nRemoving Common Keywords')
        merged_keyword_list = fea_extract.rm_common_keyword(merged_keyword_list, 0.1)
        fea_extract.print_keyword(merged_keyword_list, value_type='list')
        print('Number of %s Keywords = ' % classes[c] + str(len(merged_keyword_list)))

       # merged_keyword_list = np.array(merged_keyword_list)
       # np.save('keywords_list_'+ str(classes[c]),merged_keyword_list)
        with open('keywords_list_' + str(classes[c] + '.p'), 'wb') as fp:
            pickle.dump(merged_keyword_list, fp)

    # Generating similarity score as feature for training data
    if len(saved_train_fea[c]) > 0:
        print('\nReading features for training data')
        train_fea = saved_train_fea[c]
    else:
        print('\nGenerating similarity score as features for training data')
        train_fea = fea_extract.calc_fea(train_tweet, merged_keyword_list, num_class)
        np.save('train_fea_' + str(classes[c]), train_fea)

        for (i, tweet_content) in enumerate(train_tweet):
            print(tweet_content)
            for score in train_fea[i]:
                print "%2.4f\t" % score,
            print ""


    # Generating similarity score as feature for testing data
    if len(saved_test_fea[c]) > 0:
        print('\nReading features for testing data')
        test_fea = saved_test_fea[c]
    else:
        print('\nGenerating similarity score as features for testing data')

        test_fea = fea_extract.calc_fea(test_tweet, merged_keyword_list, num_class)
        np.save('test_fea_' + str(classes[c]), test_fea)

        for (i, tweet_content) in enumerate(test_tweet):
            print(tweet_content)
            for score in test_fea[i]:
                print "%2.4f\t" % score,
            print ""

        '''
        # Generating tf-idf as feature
        tfidf = TfidfVectorizer(max_features=15000, strip_accents='unicode', analyzer='word')#Max features-maximum number of words obtained,
        #Strip_accents=remove accents on all the words, analyzer-generate features based on words.
        tfidf.fit(train_tweet) # Learn conversion law. Apply Tf-Idf on the tweet column
        train_fea = tfidf.transform(train_tweet)  # Transform raw text documents to tfidf vectors. Learn a conversion law from documents to array data
        test_fea = tfidf.transform(test_tweet) # Apply the transformation to the test data
        '''

    if c == 0:
        all_train_fea = train_fea
        all_test_fea = test_fea
    else:
        all_train_fea = np.hstack((all_train_fea, train_fea))
        all_test_fea = np.hstack((all_test_fea, test_fea))


# Classification using Ridge Regression
all_train_mem = ip.extract_membership(train_sample, class_type='all')
all_test_mem = ip.extract_membership(test_sample, class_type='all')

'''
for i in range(len(all_train_mem)):
    all_train_mem[i] = np.array(all_train_mem[i])
for i in range(len(all_test_mem)):
    all_test_mem[i] = np.array(all_test_mem[i])

all_train_mem = sparse.csr_matrix(all_train_mem)
all_test_mem = sparse.csr_matrix(all_test_mem)
'''

test_rating = clf.linear_ridge_classify(all_test_fea, all_train_fea, all_train_mem)
print test_rating

print eval.rmse(test_rating, all_test_mem)

eval.output_csv(test_id, test_rating, 24)