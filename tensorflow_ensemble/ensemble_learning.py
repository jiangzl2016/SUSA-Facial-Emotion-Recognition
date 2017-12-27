import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import prettytensor as pt
from tensorflow_CNN.Utils import next_batch
from tensorflow_CNN.Utils import load_pkl_data

# def generate_train_validate_set():
#    return


def build_model(image_size, num_channels, num_classes):
    x = tf.placeholder(tf.float32, shape=[None, image_size * image_size], name="x")
    x_image = tf.reshape(x, [-1, image_size, image_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name="y_true")
    y_true_cls = tf.argmax(y_true, axis=1)

    # create wrapper tensor
    x_pretty = pt.wrap(x_image)
    with pt.defaults_scope(activation_fn=tf.nn.elu):
        y_pred, loss = x_pretty. \
            conv2d(5, 16, name='layer_conv1'). \
            max_pool(2, 2). \
            conv2d(3, 36, name='layer_conv2'). \
            max_pool(2, 2). \
            flatten(). \
            fully_connected(128, name='layer_fc1'). \
            softmax_classifier(num_classes=num_classes, labels=y_true)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    y_pred_cls = tf.argmax(y_pred, axis=1)
    prediction_accuracy = tf.reduct_mean(tf.cast(tf.equal(y_true_cls, y_pred_cls)
                                                 , tf.float32))
    # Define saver
    saver = tf.train.Saver()
    save_dir = './model/'
    save_path = os.path.join(save_dir, 'best_validation')

    return optimizer, x, y_true, prediction_accuracy


# early stop if last_improvement = 0 for last EARLY_STOP_ITERATIONS iterations
def run_model(num_iterations, optimizer, prediction_accuracy, x, y_true, x_train, y_train, train_batch_size=128,
              early_stop_iterations=500):
    global total_iterations
    global best_val_accuracy
    global last_improvement
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    for i in range(num_iterations):
        x_batch, y_batch = next_batch(x_train, y_train, train_batch_size)
        feed_dict = {x: x_batch, y_true: y_batch}
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 10 == 0:
            cur_accuracy = sess.run(prediction_accuracy, feed_dict=feed_dict)
            print ("Iteration: ", i + 1, " Accuracy: ", cur_accuracy)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Total time used: ", time_elapsed)


if __name__ == '__main__':
    print('Loading Data ...')
    total_iterations = 0
    best_val_accuracy = 0.0
    last_improvement = 0.0
    img_size = 48
    num_channels = 1
    num_classes = 6
    num_iterations = 100
    batch_size = 128
    #early_stop_iterations = 500
    x_train, y_train, _, _ = load_pkl_data('./imagelist.pkl')
    op, x, y_true, accuracy = build_model(img_size, num_channels,  num_classes)
    run_model(num_iterations, op, accuracy, x, y_true, x_train, y_train, batch_size)


