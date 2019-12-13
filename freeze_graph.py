# encoding:utf-8

import tensorflow as tf
import os
from lenet import LeNet


def freeze():
    if not os.path.exists('./release'):
        os.makedirs('./release')

    pb_file = './release/mnist_lenet.pb'
    ckpt_file = tf.train.latest_checkpoint('./ckpt')
    return_nodes = ['input_data','inference_label']

    net = LeNet(is_train=False)
    net.construct()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,ckpt_file)
        freeze_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    input_graph_def=sess.graph.as_graph_def(),
                                                                    output_node_names=return_nodes)

    with tf.gfile.GFile(pb_file,'wb') as f:
        f.write(freeze_graph.SerializeToString())


if __name__ == "__main__":
    freeze()

