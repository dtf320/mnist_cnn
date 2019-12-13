# encoding:utf-8


import tensorflow as tf
import numpy as np
from release_mnist import MNIST
import sys
import os


def make_tfrecord(tfrecord_path, images, labels):
    """

    :param tfrecord_path:
    :param images: [n_samples,28,28,1]
    :param labels: tuple, (n_samples,)
    :return:
    """

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    n_samples = len(labels)
    image = tf.placeholder(dtype=tf.uint8,shape=(28,28,1))
    img = tf.image.encode_png(image)
    with tf.Session("") as sess:
        for idx in range(n_samples):
            label = labels[idx]
            png_img = sess.run(img,feed_dict={image:images[idx]})
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[png_img])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
            writer.write(example.SerializeToString())
            sys.stdout.write("\r>>> Completed: %d/%d"%(idx+1,n_samples))
            sys.stdout.flush()
        writer.close()

def parse_example(example):
    features = tf.parse_single_example(
        example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.image.decode_png(features['image'], tf.uint8)
    image = tf.expand_dims(image[:,:,0]/255,axis=-1)

    label = tf.one_hot(features['label'], depth=10)
    return image, label

def load_tfrecords(tfrecord_path, repeat=-1, shuffle=100, batch_size=10):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example)
    dataset = dataset.shuffle(shuffle).repeat(repeat).batch(batch_size)
    iterator = dataset.make_one_shot_iterator().get_next()
    return iterator

def make_train_test(mnist_data_path):
    # 生成tfrecord文件
    if not os.path.exists("./tfrecord"):
        os.makedirs("./tfrecord/")
    mnist = MNIST()
    train_image, test_image, train_label, test_label = mnist.load(mnist_data_path)
    make_tfrecord("./tfrecord/train.tfrecord", train_image, train_label)
    make_tfrecord("./tfrecord/test.tfrecord", test_image, test_label)


if __name__ == '__main__':
    make_train_test('/home/datasets/mnist')

