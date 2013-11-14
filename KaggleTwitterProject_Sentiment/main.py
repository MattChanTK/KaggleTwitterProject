import csv

with open('../Data/train.csv', 'rb') as train_csv:
    train_data = csv.reader(train_csv, delimiter=',', quotechar='"')
    for row in train_data:
        print ', '.join(row)