import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
from create_sentiment_featuresets import create_feature_sets_and_labels
import numpy as np
import pickle

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
# print(train_x)
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
# do optimization in batches
batch_size = 100

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_layer_1 = {
    'f_fum': n_nodes_hl1,
    'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
}

hidden_layer_2 = {
    'f_fum': n_nodes_hl2,
    'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
}
hidden_layer_3 = {
    'f_fum': n_nodes_hl3,
    'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
}
output_layer = {
    'f_fum': None,
    'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    'biases': tf.Variable(tf.random_normal([n_classes]))
}


def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']),
                hidden_layer_1['biases'])
    # threshold function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']),
                hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']),
                hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    # one hot tensor(array)
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # compare the prediction and the known labels
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction,
        labels=y))
    #                           learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + backprop
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                # chunk through the datasets
                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                # modify those weights and biases
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1,
                  'completed out of', hm_epochs,
                  'loss:', epoch_loss)
        # asserting
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x,
                                          y: test_y}))


train_neural_network(x)
