import sklearn.feature_extraction.text as skltext
import en
import numpy as np
import spellcheck

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


# Get the keywords frequency of a text
def count_token(vectorizer, text):
    return (vectorizer.fit_transform(text)).toarray()



# Get the list of keywords
def get_keywords(vectorizer):
    return vectorizer.get_feature_names()

# Filter out bad keyword
def filter_keywords(keywords, counts):
    num_keyword = len(keywords)



    # remove keywords that are numbers
    remove_key = []
    for key in range(0, num_keyword):
        if en.is_number(keywords[key]):
            remove_key.append(key)



    # remove the associated keyword and the token counts
    counts = np.delete(counts, remove_key, 1)
    keywords = np.delete(keywords, remove_key)



    #spelling correction
    for i, word in enumerate(keywords):
        keywords[i] = spellcheck.correct(word)
    return keywords, counts

# List of important keyword and their number of occurances
def keywords_list(keywords, counts):
    counts_sum = sum(counts)
    keywords_list = dict(zip(keywords, counts_sum))
    return keywords_list

def print_keyword(keywords):
    sorted_dic = sorted(((v, k) for k, v in keywords.iteritems()), reverse=True)
    for v, k in sorted_dic:
        print "%s: %d" % (k, v)