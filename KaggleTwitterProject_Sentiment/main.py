import csv
import numpy as np


train_csv = open('../Data/train.csv', 'rb')
train_data = csv.reader(train_csv, delimiter=',', quotechar='"')

train_sample = list(train_data)


def column(matrix, i):
    return [row[i] for row in matrix]

train_s = []
for row in train_sample:
    train_s.append(row[4:9])
train_tweet = column(train_sample, 1)
train_num_sample = len(train_tweet)

for p in train_s:
    print p

for e in train_tweet:
    print e


