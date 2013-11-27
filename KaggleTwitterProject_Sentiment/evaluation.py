import math
import numpy as np

def rmse(predicted, actual):

    (num_tweet, num_class) = np.shape(actual)
    numerator = 0
    denominator = num_tweet*num_class
    for i in range(0, num_tweet):
        for j in range(0, num_class):
            numerator += (predicted[i][j] - actual[i][j])**2

    return math.sqrt(numerator/denominator)

#test code for rmse
#A = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
#B = [[0.3, 0.4, 0.3], [0.5, 0.2, 0.3]]
#print rmse(A, B)

def output_csv(id, test_prediction, num_classes, header):
    #save in the right format
    prediction = np.array(np.hstack([np.matrix(id).T, test_prediction]))#array: converts the list to array. Hstack: stack arrays horizontally
    #heading = np.array(np.hstack([header[0], header[4:28]]))
    #prediction = np.array(np.vstack([heading, prediction]))
    #matrix: creates a matrix (increase from 1 to more dimensions. T:transpose
    col = '%i,' + '%f,'* (num_classes - 1) + '%f'
    np.savetxt('prediction.csv', prediction, col, delimiter=',')