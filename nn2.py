from scipy.io import loadmat
from random import sample
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt


def nn(X,Y,test):
    t_square = np.square(test).sum(-1)
    x_square = np.square(X).sum(-1)
    tx = X.dot(test.transpose())
    norm_sq = x_square.reshape((-1,1)) - 2*tx + t_square.reshape((1,-1))
    lab_indexes = np.argmin(norm_sq, axis=0)
    preds = Y[lab_indexes]
    return preds

if __name__ == '__main__':
    start = time.time()
    ocr = loadmat('ocr.mat')
    num_trials = 10
    error_list = []
    std_list = []
    # ns = [1,10]
    ns = [ 1000, 2000, 4000, 8000 ]
    for n in ns:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            test_err[trial] = np.mean(preds != ocr['testlabels'])

        mean_error = np.mean(test_err)
        std_error = np.std(test_err)
        print("%d\t%g\t%g" % (n,mean_error,std_error))
        error_list.append(mean_error)
        std_list.append(std_error)

    err_line1 = np.array(error_list)+np.array(std_list)
    err_line2 = np.array(error_list)-np.array(std_list)
    plt.figure()
    plt.plot(ns, error_list, label="Learning Curve", color='b')
    plt.plot(ns, err_line1,label="Learning Curve + std", color='r')
    plt.plot(ns, err_line2,label="Learning Curve - std", color='g')
    plt.xlabel("n")
    plt.ylabel("Test Error")
    plt.title("Learning Curve-new")
    plt.legend(loc="upper right")
    # plt.show()
    plt.savefig("learning_curve.jpg")
    end = time.time()
    print("time taken: {}".format(end-start))

