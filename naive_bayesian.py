from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def prob(train_x, train_y):
    # 先验分布数组
    prior = np.zeros(10)
    # 条件分布矩阵
    conditional_sum = np.zeros((10, 64))  # condition probability
    for i in range(1347):
        prior[train_y[i]] = prior[train_y[i]] + 1
        for j in range(64):
            conditional_sum[train_y[i]][j] = conditional_sum[train_y[i]][j] + train_x[i][j]
    # 先验概率和条件概率
    prior_prob = (prior + 1) / 1357
    conditional_prob = (conditional_sum.T + 1) / (prior + 2)
    return prior_prob, conditional_prob

def test(test_xm, conditional_one_prob_log, conditional_zero_prob_log):
    predict_prob = np.zeros(10)
    for i in range(10):
        for j in range(64):
            predict_prob[i] += conditional_one_prob_log[j][i] \
                if test_xm[j] > 0 else conditional_zero_prob_log[j][i]
    return np.argmax(predict_prob)

def naive_bayesian(test_xm, conditional_one_prob_log, conditional_zero_prob_log):
    test_xm = np.reshape(test_xm, (450, -1))
    predict = np.zeros(450)
    for i in range(450):
        predict[i] = test(test_xm[i], conditional_one_prob_log, conditional_zero_prob_log)
    return predict


if __name__ == '__main__':
    digits = load_digits()
    data = digits.data
    train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=30)
    train_xm = (train_x > 0).astype('uint8')
    test_xm = (test_x > 0).astype('uint8')
    prior_prob, conditional_prob = prob(train_xm, train_y)
    conditional_one_prob_log = np.log(conditional_prob)
    conditional_zero_prob_log = np.log(1 - conditional_prob)
    predict_y = naive_bayesian(test_xm, conditional_one_prob_log, conditional_zero_prob_log)

    print("朴素贝叶斯分类器准确率: %.4lf" % accuracy_score(predict_y, test_y))
