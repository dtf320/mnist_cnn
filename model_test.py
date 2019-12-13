import cv2
from make_tfrecord import load_tfrecords
import numpy as np
import tensorflow as tf


def test_pb(pb_file):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    return_elements = ['input_data:0', 'inference_label:0']
    with tf.get_default_graph().as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)

    input_data = return_elements[0]
    pred_node = return_elements[1]
    iterator_test = load_tfrecords("./tfrecord/test.tfrecord", batch_size=1)
    gt = []
    pred = []
    with tf.Session() as sess:
        for i in range(10000):
            im,label = sess.run(iterator_test)
            gt.append(np.argmax(label,axis=1)[0])
            pred_label = sess.run(pred_node,feed_dict={input_data:im})
            pred.append(np.argmax(pred_label,axis=1)[0])
            print('true label {}, pred label {}'.format(np.argmax(label,axis=1),np.argmax(pred_label,axis=1)))
    acc = len([1 for i in range(len(gt)) if gt[i]==pred[i]]) / len(gt)
    print('全量mnist测试集上的准确度为 {}'.format(acc))


if __name__ == "__main__":
    test_pb('./release/mnist_lenet.pb')
