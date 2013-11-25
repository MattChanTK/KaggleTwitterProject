import sklearn.feature_extraction.text as skltext
import en
import numpy as np
import spellcheck
from nltk.stem import WordNetLemmatizer


# extract a subset of tweet with maximum membership of certain class
def extract_tweet_subset(train_tweet, train_s, class_type):

    # find the class with max membership
    train_label = []
    for i in train_s:
        train_label.append(np.argmax(i))

    # count the number of samples
    train_num_sample = len(train_tweet)

    # extract only the tweet of certain sentiment
    sub_tweet = []
    for i in range(0, train_num_sample):
        if train_label[i] == class_type:
            sub_tweet.append(train_tweet[i])

    return sub_tweet

def vectorize(min_occur=30, binary=False, min_n=1, max_n=2):

    return skltext.CountVectorizer(min_df=min_occur,  binary=binary, ngram_range=(min_n, max_n),
                                   strip_accents='unicode', lowercase=True, stop_words='english')
                                  # tokenizer=lemmatizer.LemmaTokenizer())


# Get the keywords frequency of a text
def count_token(vectorizer, text):
    return (vectorizer.fit_transform(text)).toarray()



# Get the list of keywords
def get_keywords(vectorizer):
    return vectorizer.get_feature_names()

# Filter out bad keyword
def filter_keywords(keywords, counts):

    num_keyword = len(keywords)

    remove_key = []
    for key in range(0, num_keyword):
        # remove keywords that are numbers
        if en.is_number(keywords[key]):
            remove_key.append(key)
        #remove special keywords - mention
        elif keywords[key].find('mention') != -1:
            remove_key.append(key)
        #remove special keywords - rt
        elif keywords[key].find('rt') != -1:
            remove_key.append(key)
        #remove special keywords - rt
        elif keywords[key].find('link') != -1:
            remove_key.append(key)

    # remove the associated keyword and the token counts
    counts = np.delete(counts, remove_key, 1)
    keywords = np.delete(keywords, remove_key)



    #spelling correction and lemmatizing
    lemmatizer = WordNetLemmatizer()
    for i, word in enumerate(keywords):
        keywords[i] = lemmatizer.lemmatize(spellcheck.correct(word))

    return keywords, counts

# List of important keyword and their number of occurrences
def keywords_list(keywords, counts):
    counts_sum = sum(counts)
    keywords_list = dict(zip(keywords, counts_sum))
    return keywords_list


# nicely printing the dictionary on the screen
def print_keyword(keywords, value_type='int'):

    sorted_dic = sorted(((v, k) for k, v in keywords.iteritems()), reverse=True)

    for v, k in sorted_dic:
        #left aligned with 25 chars pad
        k = '{:<25}'.format(k)
        if value_type == 'int':
            print "%s: %d" % (k, v)
        elif value_type == 'float':
            print "%s: %f" % (k, v)
        elif value_type == 'list':
            print "%s: " % k,
            for s, val in enumerate(v):
                print "%2.4f\t" % val,
            print ""
        else:
            print "Value Type Not Found!"
            break

# Merging and adding the keyword lists
def merge_keyword_lists(keyword_list):

    merged_keywords = dict()
    num_class = len(keyword_list)
    for (i, d) in enumerate(keyword_list):
        for k, v in d.iteritems():
            if k not in merged_keywords:
                merged_keywords[k] = [0]*i
                merged_keywords[k].append(v)
                for j in range(i+1, num_class):
                    merged_keywords[k].append(0)
            else:
                merged_keywords[k][i] = v

    # Padding zeros.
    return merged_keywords

#test code for the merge_keyword function
#A = {'a':1, 'b':2, 'c':3}
#B = {'b':13, 'y':23, 'x':33, 'ufo':100}
#C = {'ufo':11, 'a':20}
#print merge_keyword_lists([A,B,C])

# Normalize the keywords to find the significant score
def sig_score(merged_keywords):

    for k, v in merged_keywords.iteritems():
        sum_score = sum(v)
        for (i, score) in enumerate(v):
            merged_keywords[k][i] = score/sum_score

    return merged_keywords
