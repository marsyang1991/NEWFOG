from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model
import numpy as np
import tensorflow as tf
import sklearn
from demo import make_data_for_dnn, next_batch


class DNN:
    def __init__(self, train_xs, train_ys, validate_xs=None, validate_ys=None):
        self.train_xs = train_xs
        self.train_ys = train_ys
        if validate_xs is None or validate_ys is None:
            # if no validation set, treat the training set as validation set
            self.validate_ys = train_ys
            self.validate_xs = train_xs
        else:
            self.validate_ys = validate_ys
            self.validate_xs = validate_xs
        self.model = self.train_model()

    def train_model(self):
        model = Sequential()
        model.add(Dense(2048, activation='relu', input_dim=18 * 64))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation="softmax"))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.99, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(self.train_xs, self.train_ys,
                  epochs=1000,
                  batch_size=64, validation_data=(self.validate_xs, self.validate_ys))
        return model

    def save_model(self, file_path='model_dnn.h5'):
        self.model.save(file_path)


if __name__ == "__main__":
    data = make_data_for_dnn()
    train_xs = np.array(data['train_set'])
    train_ys = np.array(data['train_y'])
    validate_xs = np.array(data['validate_set'])
    validate_ys = np.array(data['validate_y'])
    test_xs = np.array(data['test_set'])
    test_ys = np.array(data['test_y'])
    dnn = DNN(train_xs=train_xs, train_ys=train_ys,validate_xs=validate_xs,validate_ys=validate_ys)
    dnn.save_model()
    model = dnn.model
    # score = dnn.model.evaluate(test_xs, test_ys, batch_size=64)
    # print(np.argmax(test_ys, axis=1))
    # model = load_model('model')
    score = model.evaluate(test_xs, test_ys)
    y_pred = np.argmax(model.predict(validate_xs),axis=1)
    y_true = np.argmax(validate_ys, axis=1)
    m = sklearn.metrics.confusion_matrix(y_true, y_pred)
    print(m)
    # print(score)
    # x = tf.placeholder(tf.float32, [None, 9 * 64], name='x')
    # keep_drop = 0.5
    # # 1st hidden layer
    # hidden_node = 64
    # w1 = tf.Variable(tf.zeros([9 * 64, hidden_node]), name='w1')
    # b1 = tf.Variable(tf.zeros([hidden_node]), name='b1')
    # h1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='h1')
    # h1 = tf.nn.dropout(h1, keep_prob=keep_drop)
    # # 2nd hidden layer
    # w2 = tf.Variable(tf.zeros([hidden_node, hidden_node]), name='w2')
    # b2 = tf.Variable(tf.zeros([hidden_node]), name='b2')
    # h2 = tf.nn.relu(tf.matmul(h1, w2) + b2, name='h2')
    # h2 = tf.nn.dropout(h2, keep_prob=keep_drop)
    # # 3rd hidden layer
    # w3 = tf.Variable(tf.zeros([hidden_node, hidden_node]), name='w3')
    # b3 = tf.Variable(tf.zeros([hidden_node]), name='b3')
    # h3 = tf.nn.relu(tf.matmul(h2, w3) + b3, name='h3')
    # h3 = tf.nn.dropout(h3, keep_prob=keep_drop)
    # # 4th hidden layer
    # w4 = tf.Variable(tf.zeros([hidden_node, hidden_node]), name='w4')
    # b4 = tf.Variable(tf.zeros([hidden_node]), name='b4')
    # h4 = tf.nn.relu(tf.matmul(h3, w4) + b4, name='h4')
    # h4 = tf.nn.dropout(h4, keep_prob=keep_drop)
    # # 5th hidden layer
    # w5 = tf.Variable(tf.zeros([hidden_node, hidden_node]), name='w5')
    # b5 = tf.Variable(tf.zeros([hidden_node]), name='b5')
    # h5 = tf.nn.relu(tf.matmul(h4, w5) + b5, name='h5')
    # h5 = tf.nn.dropout(h5, keep_prob=keep_drop)
    # # output layer
    # w6 = tf.Variable(tf.zeros([hidden_node, 2]), name='w6')
    # b6 = tf.Variable(tf.zeros([2]), name='b5')
    # y = tf.nn.softmax(tf.matmul(h5, w6) + b6, name='y_pred')
    #
    # y_ = tf.placeholder(tf.float32, [None, 2], name='y_true')
    #
    # # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.log(y)))
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # # writer = tf.summary.FileWriter('log', tf.get_default_graph())
    # # writer.close()
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # # y_pred = tf.reduce_sum(tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # init = tf.global_variables_initializer()
    # session = tf.Session()
    # session.run(init)
    # batch = 64
    # for i in range(100):
    #     # batch_xs = train_xs[64 * i:64 * (i + 1)]
    #     # batch_ys = train_ys[64 * i:64 * (i + 1)]
    #     batch_xs, batch_ys = next_batch(train_xs, train_ys, i, batch)
    #     session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #     # acc_vali = session.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    #     if i % 50 == 0:
    #         # print(session.run(y, feed_dict={x: batch_xs, y_: batch_ys}))
    #         print(session.run(y, feed_dict={x: validate_xs, y_: validate_ys}))
    #         # print(acc_vali)
    # session.close()
