# -*- coding:utf-8 -*-

__author__ = 'Duan Tengfei'

import tensorflow as tf
from make_tfrecord import load_tfrecords
import time


class LeNet:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def construct(self,lr=None):
        if self.is_train and (lr is None):
            raise Exception('在训练模式下必须输入学习率lr')

        # 1 卷积层
        # x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
        self.image = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_data")
        shape = [5, 5, 1, 32]
        W_conv1 = tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="default")
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
        # 2 卷积层
        shape = [5, 5, 32, 64]
        W_conv2 = tf.Variable(tf.truncated_normal(shape,stddev=0.1),name="default")
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
        # 3 全连接层
        W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024],stddev=0.1),name="default")
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # 4 Dropout
        if self.is_train:
            self.keep_prob = tf.placeholder("float", name="keep_prob")
        else:
            self.keep_prob = tf.constant(1,dtype=tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # 5 Softmax输出层
        W_fc2 = tf.Variable(tf.truncated_normal([1024, 10],stddev = 0.1),name="default")
        b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]))
        soft_in = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.y_conv = tf.nn.softmax(soft_in, name="inference_label")

        if self.is_train:
            self.y_true = tf.placeholder("float", [None, 10], name="true_label")
            self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_true * tf.log(self.y_conv+1e-16)))
            tf.summary.scalar("loss", self.cross_entropy)
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.cross_entropy)
            self.correct_prediction = tf.equal(tf.argmax(self.y_conv, axis=1), tf.argmax(self.y_true, axis=1))
            self.predict_label = tf.argmax(self.y_conv, axis=1)
            self.predict_label_onehot = self.y_conv
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"), name="accuracy_ops")
            tf.summary.scalar("train-accuracy", self.accuracy)

    def train(self, epoch, batch_size,lr=None):
        if not self.is_train:
            raise Exception('不允许在self.is_train=False情况下训练网络')

        self.construct(lr)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        TIMESTAMP = time.strftime("%Y-%m-%d--%H-%M-%S")
        writer = tf.summary.FileWriter("./logs/"+TIMESTAMP,self.sess.graph)
        saver = tf.train.Saver()

        MAX_ITER = epoch*60000//batch_size
        iterator_train = load_tfrecords("./tfrecord/train.tfrecord",batch_size=batch_size)
        iterator_test = load_tfrecords("./tfrecord/test.tfrecord",batch_size=500)
        for iter in range(1,MAX_ITER+1):
            image_train, label_train = self.sess.run(iterator_train)
            image_test, label_test = self.sess.run(iterator_test)
            train_accuracy = self.sess.run(self.accuracy,feed_dict={self.image: image_train,
                                                                    self.y_true: label_train,
                                                                    self.keep_prob: 1.0})
            cross_entropy = self.sess.run(self.cross_entropy,feed_dict={self.image: image_train,
                                                                        self.y_true: label_train,
                                                                        self.keep_prob: 1.0})
            test_accuracy = self.sess.run(self.accuracy,feed_dict={self.image: image_test,
                                                                   self.y_true: label_test,
                                                                   self.keep_prob: 1.0})
            self.sess.run(self.train_step, feed_dict = {self.image: image_train,
                                                        self.y_true: label_train,
                                                        self.keep_prob: 0.5})
            print("iter %d  train accuracy %f  loss %f  test accuracy %f" % (iter,train_accuracy,cross_entropy,test_accuracy))

            if iter % 10 == 0:
                saver.save(self.sess,save_path="./ckpt/mnist-lenet",global_step=iter)
                ret = self.sess.run(merged,feed_dict={self.image: image_train,
                                                      self.y_true: label_train,
                                                      self.keep_prob: 1.0})
                writer.add_summary(ret,global_step=iter)


if __name__ == "__main__":
    cnn = LeNet()
    cnn.train(epoch=15,batch_size=500,lr=0.001)