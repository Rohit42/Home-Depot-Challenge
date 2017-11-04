import cv2
import numpy as np
import pandas
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import csv
import math
import matplotlib.pyplot as plt


def create_placeholders(n_H0, n_W0, n_C0, n_y):  # n_H0 = 65, n_W0 = 65, n_C0 = 3, n_y = 6
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))  # n_y = 6

    return X, Y


def initialize_parameters():
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 4, 4, 1], padding="SAME")
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding="SAME")
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    dropout = tf.layers.dropout(
        inputs=Z3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    return Z3


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X_train.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.004, num_epochs=200, minibatch_size=64, print_cost=True):
    # ops.reset_default_graph()
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    accs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                predict_op = tf.argmax(Z3, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                costs.append(minibatch_cost)
                accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
                accs.append(accuracy)

        # plot the cost
        plt.plot(np.squeeze(accs))
        plt.ylabel('acc')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


def load_data():
    X_train = []
    Y_train = []
    wordsToNums = {'Chandeliers': 0, 'Showerheads': 1, 'Ceiling Fans': 2, 'Vanity Lighting': 3, 'Floor Lamps': 4,
                   'Single Handle Bathroom Sink Faucets': 5}
    numsToWords = {0: 'Chandeliers', 1: 'Showerheads', 2: 'Ceiling Fans', 3: 'Vanity Lighting', 4: 'Floor Lamps',
                   5: 'Single Handle Bathroom Sink Faucets'}
    idToCat = {}
    with open('data/Xy_train.txt') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for row in reader:
            idToCat[row['itemId']] = row['category']

    for file in os.listdir("data/train"):
        newImg = cv2.imread(os.path.join("data/train", file))
        num = file[: file.find(".")]
        cat = idToCat[num]
        Y_train.append(wordsToNums[cat])
        X_train.append(newImg)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    # indices = np.array([i for i in range(len(X_train))])
    # np.random.shuffle(indices)
    # X_train = X_train[indices]
    # Y_train = Y_train[indices]
    Y_train = np.eye(6)[Y_train.reshape(-1)]
    m = X_train.shape[0]
    X_test = X_train[:3 * m // 4, :, :, :]
    X_train = X_train[int(3 / 4 * m):, :, :, :]
    Y_test = Y_train[:int(3 / 4 * m), :]
    Y_train = Y_train[int(3 / 4 * m):, :]
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data()
print(X_train.shape)
trAcc, teAcc, parameters = model(X_train, Y_train, X_test, Y_test)