from sklearn import linear_model
import numpy as np

def linear_ridge_classify(test_fea, train_fea, gnd, alpha=0.1):
    clf = linear_model.Ridge(alpha=alpha, solver='sparse_cg')
    train_fea = np.array(train_fea)
    gnd = np.array(gnd)
    #print train_fea[10]
    #print gnd[10]
    clf.fit(train_fea, gnd)
    test_prediction = clf.predict(test_fea)

    #cleaning negative values
    test_prediction = test_prediction.clip(0.0,1.0)

    return test_prediction
