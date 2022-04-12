#B711093 성의현
import KNN_class
import numpy as np
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

#데이터 불러오기
#train은 학습용 데이터로, 6만개가 있으며, x는 속성이 784개, t는 label이 저장되어있다.
#test는 학습 확인용 데이터로 1만개가 있다. x와 t구분은 train과 동일하다.
(x_train, train_data_label), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#확인용 데이터를 일부만 저장해서 확인하기 위한 배열이다.
train_data = np.empty((0,28), dtype='int32')
test_data = np.empty((0, 28), dtype='int32')
test_data_label = np.array([], dtype='int32')
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#연산의 횟수를 줄이기 위해서 데이터를 가공한다. 학습용 데이터를 모두 가공한다.
for i in range(0, len(x_train)) :
    image = x_train[i].reshape(28,28)
    tmp = image.sum(axis=1)
    train_data = np.append(train_data, np.array([tmp]), axis=0)
print("train data ready...")

#테스트용 데이터는 집어넣으면서 가공한다.
size = 100 #확인할 데이터의 수
sample = np.random.randint(0, t_test.shape[0], size)
for i in sample :
    image = x_test[i].reshape(28,28)
    tmp = image.sum(axis=1)
    test_data = np.append(test_data, np.array([tmp]), axis=0)
    test_data_label = np.append(test_data_label, np.array([t_test[i]]), axis=0)
print("ready to start...")

#KNN알고리즘을 이용한 classification
mnist_KNN = KNN_class.KNN(train_data, train_data_label, label_names, 15)
correct_count = 0
for i in range(0, len(test_data)) :
    result = mnist_KNN.test(test_data[i], 'w_vote')
    if (result == label_names[test_data_label[i]]) : correct_count += 1
    print("Test Data label>> " + str(test_data_label[i]) + "  KNN Result>> " + result)
print("accuracy>> "  + str(correct_count / size))
