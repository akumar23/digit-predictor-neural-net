import neural_net
import training
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

W1, b1, W2, b2 = training.gradient_descent(X_train, Y_train, 0.10, 500, m)

try:
    while True:
        print("Press Ctrl+C to stop.")
        pred_index = input("Enter which prediction from the training set you want to see: ")
        neural_net.test_prediction(X_train, Y_train, int(pred_index), W1, b1, W2, b2)
except KeyboardInterrupt:
    print("Program stopped by user.")