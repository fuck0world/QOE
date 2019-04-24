from collections import Counter
from imblearn.over_sampling import SMOTE
import sys 
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from collections import Counter
from sklearn.metrics import  accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generatebatch(X,Y,n_examples, batch_size): 
    for batch_i in range(n_examples // batch_size): 
        start = batch_i * batch_size 
        end = start + batch_size 
        batch_xs = X[start:end] 
        batch_ys = Y[start:end] 
        yield batch_xs, batch_ys 
    
data = pd.read_csv('E:\\备份\\训练数据汇总(8W)\\2019_4_15.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

name = ['PHONE_VERSION', 'VIDEO_CLARITY']
columns = data.columns.values.tolist()
name_1 = [name for index, name in enumerate(columns) if name != name[0] and name != name[1]]

scaler = MinMaxScaler()
X1 = data[name_1[0:16]]
X1_data = scaler.fit_transform(X1)
Y = data[name_1[16:21]]
X2 = data[name[0]]
X3 = data[name[1]]

X2_data = OneHotEncoder().fit_transform(X2.values.reshape(-1, 1)).todense().getA()
X3_data = OneHotEncoder().fit_transform(X3.values.reshape(-1, 1)).todense().getA()

X_data = np.hstack((X1_data, X2_data, X3_data))

Y1 = data[[name_1[16]]]
Y2 = data[[name_1[17]]]
Y3 = data[[name_1[18]]]
Y4 = data[[name_1[19]]]

smo = SMOTE(random_state=42)
X1_smo, y1_smo = smo.fit_sample(X_data, Y1)

smo = SMOTE(random_state=42)
X2_smo, y2_smo = smo.fit_sample(X_data, Y2)

smo = SMOTE(random_state=42)
X3_smo, y3_smo = smo.fit_sample(X_data, Y3)

smo = SMOTE(random_state=42)
X4_smo, y4_smo = smo.fit_sample(X_data, Y4)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1_smo, y1_smo, 
                                                        test_size=0.25, 
                                                        random_state = 33)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2_smo, y2_smo, 
                                                        test_size=0.25, 
                                                        random_state = 33)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3_smo, y3_smo, 
                                                        test_size=0.25, 
                                                        random_state = 33)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4_smo, y4_smo, 
                                                        test_size=0.25, 
                                                        random_state = 33)

y1_train = OneHotEncoder().fit_transform(y1_train.reshape(-1, 1)).todense().getA()
y1_test_code = OneHotEncoder().fit_transform(y1_test.reshape(-1, 1)).todense().getA()
y2_train = OneHotEncoder().fit_transform(y2_train.reshape(-1, 1)).todense().getA()
y2_test_code = OneHotEncoder().fit_transform(y2_test.reshape(-1, 1)).todense().getA()
y3_train = OneHotEncoder().fit_transform(y3_train.reshape(-1, 1)).todense().getA()
y3_test_code = OneHotEncoder().fit_transform(y3_test.reshape(-1, 1)).todense().getA()
y4_train = OneHotEncoder().fit_transform(y4_train.reshape(-1, 1)).todense().getA()
y4_test_code = OneHotEncoder().fit_transform(y4_test.reshape(-1, 1)).todense().getA()

def compute_accuracy(v_x, v_y):
    global pred
    #input v_x to nn and get the result with y_pre
    y_pre = sess.run(pred, feed_dict={x:v_x})
    #find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_y,1))
    #calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #get input content
    result = sess.run(accuracy,feed_dict={x: v_x, y: v_y})
    return result

def Bi_lstm(X):
    lstm_f_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_b_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    return tf.contrib.rnn.static_bidirectional_rnn(lstm_f_cell, lstm_b_cell, X, dtype=tf.float32)

def RNN(X,weights,biases):
    # hidden layer for input
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']

    #reshape data put into bi-lstm cell
    X_in = tf.reshape(X_in, [-1,n_steps, n_hidden_units])
    X_in = tf.transpose(X_in, [1,0,2])
    X_in = tf.reshape(X_in, [-1, n_hidden_units])
    X_in = tf.split(X_in, n_steps)
    outputs, _, _ = Bi_lstm(X_in)
    
    #hidden layer for output as the final results
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results
    
# parameters init
l_r = 0.001
training_iters = 1000
batch_size = 128

n_inputs = 7
n_steps = 4
n_hidden_units = 128
n_classes = 5

#define placeholder for input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define w and b
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out': tf.Variable(tf.random_normal([2*n_hidden_units,n_classes]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out': tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(l_r).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#init session
sess = tf.Session()
#init all variables
sess.run(tf.global_variables_initializer())
print("######### The train & test process of SCORE 1 ##########")
print("开始时间:",datetime.datetime.now())
for i in range(1000):
    for batch_xs,batch_ys in generatebatch(X1_train, y1_train, y1_train.shape[0], 
                                           batch_size):
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op,feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        print(sess.run(accuracy,feed_dict={x: batch_xs, y: batch_ys,}))
print("结束时间:",datetime.datetime.now())
test_data = X1_test.reshape([-1, n_steps, n_inputs])
test_label = y1_test_code
print("Testing Accuracy: ", compute_accuracy(test_data, test_label))
y1_pre = sess.run(pred, feed_dict={x:test_data})
a1 = y1_pre.argmax(axis = 1)
b1 = a1 + 1
print("测试集数据分布：")
print(Counter(b1.tolist()))

#init session
sess = tf.Session()
#init all variables
sess.run(tf.global_variables_initializer())
print("######### The train & test process of SCORE 2 ##########")
print("开始时间:",datetime.datetime.now())
for i in range(1000):
    for batch_xs,batch_ys in generatebatch(X2_train, y2_train, y2_train.shape[0], 
                                           batch_size):
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op,feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        print(sess.run(accuracy,feed_dict={x: batch_xs, y: batch_ys,}))
print("结束时间:",datetime.datetime.now())
test_data = X2_test.reshape([-1, n_steps, n_inputs])
test_label = y2_test_code
print("Testing Accuracy: ", compute_accuracy(test_data, test_label))
y2_pre = sess.run(pred, feed_dict={x:test_data})
a2 = y2_pre.argmax(axis = 1)
b2 = a2 + 1
print("测试集数据分布：")
print(Counter(b2.tolist()))

#init session
sess = tf.Session()
#init all variables
sess.run(tf.global_variables_initializer())
print("######### The train & test process of SCORE 3 ##########")
print("开始时间:",datetime.datetime.now())
for i in range(1000):
    for batch_xs,batch_ys in generatebatch(X3_train, y3_train, y3_train.shape[0], 
                                           batch_size):
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op,feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        print(sess.run(accuracy,feed_dict={x: batch_xs, y: batch_ys,}))
print("结束时间:",datetime.datetime.now())
test_data = X3_test.reshape([-1, n_steps, n_inputs])
test_label = y3_test_code
print("Testing Accuracy: ", compute_accuracy(test_data, test_label))
y3_pre = sess.run(pred, feed_dict={x:test_data})
a3 = y3_pre.argmax(axis = 1)
b3 = a3 + 1
print("测试集数据分布：")
print(Counter(b3.tolist()))

#init session
sess = tf.Session()
#init all variables
sess.run(tf.global_variables_initializer())
print("######### The train & test process of SCORE 4 ##########")
print("开始时间:",datetime.datetime.now())
for i in range(1000):
    for batch_xs,batch_ys in generatebatch(X4_train, y4_train, y4_train.shape[0], 
                                           batch_size):
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_op,feed_dict={x: batch_xs, y: batch_ys})
    if i % 100 == 0:
        print(sess.run(accuracy,feed_dict={x: batch_xs, y: batch_ys,}))
print("结束时间:",datetime.datetime.now())
test_data = X4_test.reshape([-1, n_steps, n_inputs])
test_label = y4_test_code
print("Testing Accuracy: ", compute_accuracy(test_data, test_label))
y4_pre = sess.run(pred, feed_dict={x:test_data})
a4 = y4_pre.argmax(axis = 1)
b4 = a4 + 1
print("测试集数据分布：")
print(Counter(b4.tolist()))