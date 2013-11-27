import csv

def column(matrix, i):
    return [row[i] for row in matrix]


def import_csv(path):
    train_csv = open(path, 'rb')
    train_data = csv.reader(train_csv, delimiter=',', quotechar='"')

    train_sample = tuple(train_data)

    return train_sample
def extract_header(train_sample):
    return train_sample[0]

def extract_tweet_id(train_sample):
    # extract the tweet strings
    train_tweet = column(train_sample, 0)
    if train_tweet:
        # remove the heading
        train_tweet.pop(0)

    return train_tweet


def extract_tweet(train_sample):
    # extract the tweet strings
    train_tweet = column(train_sample, 1)
    if train_tweet:
        # remove the heading
        train_tweet.pop(0)

    return train_tweet


def extract_membership(train_sample, class_type='s'):


    if class_type == 'w':
        set_i = 9
        set_f = 13
    elif class_type == 'c':
        set_i = 13
        set_f = 28
    elif class_type == 'all':
        set_i = 4
        set_f = 28
    elif class_type == 'sw':
        set_i = 4
        set_f = 13
    else:
        set_i = 4
        set_f = 9

    train_s = []
    if train_sample:
        train_num_sample = len(train_sample) - 1
        # extract the sentiment counts
        for row in train_sample:
            train_s.append(row[set_i:set_f])
        if train_s:
            # remove the heading
            train_s.pop(0)
            #count the number of class
            num_class = len(train_s[0])
            #convert the labels to float
            for i in range(0, train_num_sample):
                for j in range(0, num_class):
                    train_s[i][j] = float(train_s[i][j])

    return train_s






