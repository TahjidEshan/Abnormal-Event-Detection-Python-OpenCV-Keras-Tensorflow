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
from sklearn.naive_bayes import GaussianNB
import sys
import pickle


def svm(training_features, training_labels, test_features, test_labels):
    print("Initiating SVM")
    model = OneVsRestClassifier(estimator=SVC(
        kernel='poly', degree=2, random_state=0))
    model.fit(training_features, training_labels)
    print "SVM Accuracy:", (model.score(test_features, test_labels))
    filename = 'trained_SVM.sav'
    print('Saving Model')
    pickle.dump(model, open(filename, 'wb'))
    return None


def naiveBayes(training_features, training_labels, test_features, test_labels):
    print("Initiating Naive Bayes")
    model = GaussianNB()
    model.fit(training_features, training_labels)
    print "Naive Bayes Accuracy:", (model.score(test_features, test_labels))
    filename = 'trained_Naive_Bayes.sav'
    print('Saving Model')
    pickle.dump(model, open(filename, 'wb'))
    return None


def logReg(training_features, training_labels, test_features, test_labels, learning_rate, training_epochs, X, Y, W):
    print("Initiating Logistics Regression")
    init = tf.initialize_all_variables()
    y_ = tf.nn.sigmoid(tf.matmul(X, W))
    cost_function = tf.reduce_mean(tf.reduce_sum(
        (-Y * tf.log(y_)) - ((1 - Y) * tf.log(1 - y_)), reduction_indices=[1]))
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(cost_function)
    cost_history = np.empty(shape=[1], dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={
                     X: training_features, Y: training_labels})
            cost_history = np.append(cost_history, sess.run(
                cost_function, feed_dict={X: training_features, Y: training_labels}))
        y_pred = sess.run(y_, feed_dict={X: test_features})
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print "Logistics Regression Accuracy: ", (sess.run(accuracy, feed_dict={X: test_features, Y: test_labels}))
        saver = tf.train.Saver()
        print('Saving Model')
        saver.save(sess, 'logistics_regression')
        saver.export_meta_graph('logistics_regression.meta')
    return None


def neuralNetKeras(training_data, training_labels, test_data, test_labels, n_dim):
    print("Initiating Neural Network")
    seed = 8
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(128, input_dim=n_dim, init='uniform', activation='relu'))
    model.add(Dense(64, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(training_data, training_labels, nb_epoch=30, batch_size=10)
    scores = model.evaluate(test_data, test_labels)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    #filename = 'trained_Neural_Net.sav'
    print('Saving Model')
    model.save('trained_neural_net.h5')
    #pickle.dump(model, open(filename, 'wb'))
    return None


def cnnKeras(training_data, training_labels, test_data, test_labels, n_dim):
    print("Initiating CNN")
    seed = 8
    np.random.seed(seed)
    num_classes = 2
    model = Sequential()
    model.add(Convolution2D(64, 1, 1, init='glorot_uniform', border_mode='valid',
                            input_shape=(2, 2000, 1500), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Convolution2D(32, 1, 1, init='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(training_data, training_labels, validation_data=(
        test_data, test_labels), nb_epoch=30, batch_size=8, verbose=2)

    scores = model.evaluate(test_data, test_labels, verbose=1)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
    print('Saving Model')
    model.save('trained_CNN.h5')
    #filename = 'trained_CNN.sav'
    #pickle.dump(model, open(filename, 'wb'))
    return None


def reshapeList(features):
    x = len(features)
    y = len(features[0][0]) * \
        len(features[0][0][0]) * len(features[0])
    features = np.reshape(features, (x, y))
    return np.array(features)


def reshapeData(features):
    return np.reshape(features, (len(features), len(features[0]), len(features[0][0]), len(features[0][0][0])))


def readData(dataList):
    featureList = []
    for i in range(0, len(dataList.index)):
        temp = []
        features = ast.literal_eval(dataList.loc[i, 'features'])
        pixels = ast.literal_eval(dataList.loc[i, 'pixels'])
        temp.append(np.array(features))
        temp.append(np.array(pixels))
        featureList.append(np.array(temp))
    return featureList


def main():

    data0 = pd.read_csv('data.csv')
    data1 = pd.read_csv('data1.csv')
    #frames = [data0, data1]
    #data = pd.concat(frames)
    # shuffle the data
    # print(list(data.columns))
    #print('Shuffling Data')
    data0 = data0.sample(frac=1).reset_index(drop=True)
    data1 = data1.sample(frac=1).reset_index(drop=True)

    # Calculate length of 30% data for testing
    val_text = len(data0.index) * (30.0 / 100.0)
    val1_text = len(data1.index) * (30.0 / 100.0)

    # divide training and test data
    test_data0 = data0.tail(int(val_text)).reset_index(drop=True)
    training_data0 = data0.head(len(data0.index) - int(val_text))
    test_data1 = data1.tail(int(val1_text)).reset_index(drop=True)
    training_data1 = data1.head(len(data1.index) - int(val1_text))
    #test_data = data.tail(1).reset_index(drop=True)
    #training_data = data.head(1)

    # set labels
    # labels for svm
    # print(list(training_data.columns))
    training_labels0 = training_data0['labels'].values
    test_labels0 = test_data0['labels'].values
    training_labels1 = training_data1['labels'].values
    test_labels1 = test_data1['labels'].values
    # labels for tf classifiers

    training_features_final0 = []
    test_features_final0 = []
    training_features_final1 = []
    test_features_final1 = []

    print("Reading Data")
    training_features_final0 = readData(training_data0)
    test_features_final0 = readData(test_data0)
    training_features_final1 = readData(training_data1)
    test_features_final1 = readData(test_data1)

    training_features_final = np.concatenate(
        (training_features_final0, training_features_final1), axis=0)
    test_features_final = np.concatenate(
        (test_features_final0, test_features_final1), axis=0)
    training_labels = np.concatenate(
        (training_labels0, training_labels1), axis=0)
    test_labels = np.concatenate(
        (test_labels0, test_labels1), axis=0)
    training_X = np.array(training_labels)
    test_X = np.array(test_labels)
    training_X = np.reshape(training_X, (len(training_X), 1))
    test_X = np.reshape(test_X, (len(test_X), 1))

    print("Reshaping Data")
    train_cnn = reshapeData(training_features_final)
    test_cnn = reshapeData(test_features_final)
    training_features_final = reshapeList(training_features_final)
    test_features_final = reshapeList(test_features_final)
    n_dim = training_features_final.shape[1]

    while(True):

        print('Please choose any one of the options:')
        print('Press 1 for Support Vector Machine')
        print('Press 2 for Logistics Regression')
        print('Press 3 for Naive Bayes Classifier')
        print('Press 4 for Neural Network')
        print('Press 5 for Convolutional Neural Network')
        print('Press 6 to Exit')
        print('Press 7 for all')

        choice = int(raw_input())
        bool = False

        if choice == 6:
            break

        if choice == 1 or choice == 7:
            '''
            ###SVM classifier using sklearn
            '''
            bool = True
            svm(training_features_final, training_labels,
                test_features_final, test_labels)

            '''
            ###Logistics Regression Using tensorflow
            '''
        if choice == 2 or choice == 7:
            bool = True
            #print("Initiating Logistics Regression")
            learning_rate = 0.1
            training_epochs = 28

            X = tf.placeholder(tf.float32, [None, n_dim])
            Y = tf.placeholder(tf.float32, [None, 1])
            W = tf.Variable(tf.ones([n_dim, 2]))

            logReg(training_features_final, training_X, test_features_final,
                   test_X, learning_rate, training_epochs, X, Y, W)

        if choice == 4 or choice == 7:
            bool = True
            #print("Initialising Neural Network")
            neuralNetKeras(training_features_final, training_X,
                           test_features_final, test_labels, n_dim)

        if choice == 5 or choice == 7:
            bool = True
            #print("Initialising convolutional neural network ")
            cnnKeras(train_cnn, training_X,
                     test_cnn, test_X, n_dim)

        if choice == 3 or choice == 7:
            bool = True
            #print("Initialising Naive Bayes")
            naiveBayes(training_features_final, training_labels,
                       test_features_final, test_labels)

        if bool == False:
            print('Wrong Choice, Please Choose Again')

        if choice == 7:
            break

    sys.exit()

if __name__ == '__main__':
    main()
