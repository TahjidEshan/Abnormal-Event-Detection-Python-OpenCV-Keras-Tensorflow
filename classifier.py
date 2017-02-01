import pandas as pd
from sklearn import svm
import numpy as np
import ast
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def svm(training_features, training_labels, test_features, test_labels):
    model = OneVsRestClassifier(estimator=SVC(
        kernel='poly', degree=2, random_state=0))
    model.fit(training_features, training_labels)
    print "Accuracy:", (model.score(test_features, test_labels))


def logReg(training_features, training_labels, test_features, test_labels, learning_rate, training_epochs, X, Y, W):
    init = tf.initialize_all_variables()
    y_ = tf.nn.sigmoid(tf.matmul(X, W))
    cost_function = tf.reduce_mean(tf.reduce_sum((-Y * tf.log(y_)) - ((1 - Y) * tf.log(1 - y_)),
                                                 reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(cost_function)
    cost_history = np.empty(shape=[1], dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={
                     X: training_features, Y: training_labels})
            cost_history = np.append(cost_history, sess.run(cost_function,
                                                            feed_dict={X: training_features, Y: training_labels}))

        y_pred = sess.run(y_, feed_dict={X: test_features})
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print "Accuracy: ", (sess.run(accuracy, feed_dict={X: test_features, Y: test_labels}))


def batch_creator(batch_size, train_x, train_y, seed):
    """Create batch with random samples and return appropriate format"""
    start = np.randint(0, seed)
    end = start + batch_size
    return train_x[start:end], train_y[start:end]


def cnn(training_features, training_labels, test_features, test_labels, learning_rate, training_epochs, n_dim, x, y):
    #input_num_units = n_dim
    #hidden_num_units = 500
    input_num_units = 12
    hidden_num_units = 8

    output_num_units = 1
    batch_size = 4
    seed = len(training_features) - batch_size
    weights = {
        'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
    }
    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)

    output_layer = tf.matmul(hidden_layer, weights[
                             'output']) + biases['output']
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    with tf.Session("/cpu:0") as sess:
        # create initialized variables
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(training_features.shape[0] / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(
                    batch_size, training_features, training_labels, seed)
                _, c = sess.run([optimizer, cost], feed_dict={
                                x: batch_x, y: batch_y})

                avg_cost += c / total_batch

            print "Epoch:", (epoch + 1), "cost =", "{:.5f}".format(avg_cost)

        print "\nTraining complete!"

        # find predictions on val set
        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print "Accuracy: ", (sess.run(accuracy, feed_dict={X: test_features, Y: test_labels}))

        predict = tf.argmax(output_layer, 1)
        pred = predict.eval({x: test_features.reshape(-1, input_num_units)})


def neuralNetKeras(training_data, training_labels, test_data, test_labels, n_dim):
    seed = 8
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(12, input_dim=n_dim, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(training_data, training_labels, nb_epoch=150, batch_size=10)
    scores = model.evaluate(test_data, test_labels)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def cnnKeras(training_data, training_labels, test_data, test_labels, n_dim):
    seed = 8
    np.random.seed(seed)
    num_classes = 2
    model = Sequential()
    model.add(Convolution2D(30, 1, 6000000, border_mode='valid',
                            input_shape=(1, n_dim, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Convolution2D(15, 1, 6000000, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(training_data, training_labels, validation_data=(
        test_data, test_labels), nb_epoch=10, batch_size=8, verbose=2)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


def reshapeList(features):
    x = len(features)
    y = len(features[0][0]) * \
        len(features[0][0][0]) * len(features[0])
    features = np.reshape(
        features, (x, y))
    return np.array(features)


def readData(featureList, dataList):
    for i in range(0, len(dataList.index)):
        temp = []
        features = ast.literal_eval(dataList.loc[i, 'features'])
        pixels = ast.literal_eval(dataList.loc[i, 'pixels'])
        temp.append(np.array(features))
        temp.append(np.array(pixels))
        featureList.append(np.array(temp))
    return featureList


def main():

    data = pd.read_csv('data.csv')
    # shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Calculate length of 30% data for testing
    val_text = len(data.index) * (30.0 / 100.0)

    # divide training and test data
    test_data = data.tail(int(val_text)).reset_index(drop=True)
    training_data = data.head(len(data.index) - int(val_text))
    #test_data = data.tail(1).reset_index(drop=True)
    #training_data = data.head(2)

    # set labels
    # labels for svm
    training_labels = training_data['labels'].values
    test_labels = test_data['labels'].values
    # labels for tf classifiers
    training_X = np.array(training_labels)
    test_X = np.array(test_labels)
    training_X = np.reshape(training_X, (len(training_X), 1))
    test_X = np.reshape(test_X, (len(test_X), 1))

    training_features_final = []
    test_features_final = []

    print("Reading Data")
    training_features_final = readData(training_features_final, training_data)
    test_features_final = readData(test_features_final, test_data)

    print("Reshaping Data")
    training_features_final = reshapeList(training_features_final)
    test_features_final = reshapeList(test_features_final)

    '''
    ###SVM classifier using sklearn
    '''
    '''
    print("Initiating SVM")
    svm(training_features_final, training_labels,
        test_features_final, test_labels)
    '''
    '''
    ###Logistics Regression Using tensorflow
    '''
    #print("Initiating Logistics Regression")
    n_dim = training_features_final.shape[1]
    learning_rate = 0.1
    training_epochs = 10

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.ones([n_dim, 2]))
    '''
    logReg(training_features_final, training_X, test_features_final,
           test_X, learning_rate, training_epochs, X, Y, W)
    '''
    # Starting CNN
    '''
    print("Starting Convolutional Neural Network")
    cnn(training_features_final, training_X, test_features_final,
        test_X, learning_rate, training_epochs, n_dim, X, Y)
    '''
    '''
    print("Initialising Neural Network")
    neuralNetKeras(training_features_final, training_X,
                   test_features_final, test_labels, n_dim)
    '''
    print("Initialising convolutional neural network ")
    cnnKeras(training_features_final, training_X,
             test_features_final, test_labels, n_dim)

if __name__ == '__main__':
    main()
