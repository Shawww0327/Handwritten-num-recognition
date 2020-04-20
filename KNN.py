from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np


# 计算测试数据与每一个训练数据的距离
def distance(train_x, test_x, i):
    all_dist = np.sqrt(np.sum(np.square(test_x[i] - train_x), axis=1)).tolist()
    return all_dist

# 查找最近的K个数据所对应的预测值
def get_number(all_dist):
    all_number = []
    for i in range(5):
        min_index = np.argmin(all_dist)
        number = (train_y[min_index]).tolist()
        all_number.append(number)
        #在距离数组中，将最小的距离值删去
        min_number = min(all_dist)
        min_number_index = all_dist.index(min_number)
        del all_dist[min_number_index]
    return all_number

# 通过众数找到频率最多的类
def get_most_number(all_number):
    new_number = np.array(all_number)
    counts = np.bincount(new_number)
    return np.argmax(counts)

def knn(train_x, test_x, test_y):
    predict_y = []
    for i in range(len(test_x)):
        all_dist = distance(train_x, test_x, i)
        all_number = get_number(all_dist)
        min_number = get_most_number(all_number)
        predict_y.append(min_number)
    print("KNN决策树准确率: %.4lf" % accuracy_score(predict_y, test_y))

if __name__ == '__main__':
    digits = load_digits()
    train_x, test_x, train_y, test_y = train_test_split(digits.data, digits.target, test_size=0.25, random_state=30)
    ss = preprocessing.StandardScaler()
    train_xs = ss.fit_transform(train_x)
    test_xs = ss.transform(test_x)
    knn(train_xs, test_xs, test_y)
