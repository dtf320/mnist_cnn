import cv2
from make_tfrecord import load_tfrecords
import numpy as np
import tensorflow as tf


def display(pb_file):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    return_elements = ['input_data:0', 'inference_label:0']
    with tf.get_default_graph().as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)

    input_data = return_elements[0]
    pred = return_elements[1]
    iterator_test = load_tfrecords("./tfrecord/test.tfrecord", batch_size=1)
    with tf.Session() as sess:
        for i in range(10000):
            im,label = sess.run(iterator_test)
            im_display = (255*im[0,...]).astype(np.uint8)
            pred_label = sess.run(pred,feed_dict={input_data:im})
            print('true label {}, pred label {}'.format(np.argmax(label,axis=1),np.argmax(pred_label,axis=1)))
            cv2.imshow('',im_display)
            if cv2.waitKey(0) == 27:      # 回车显示下一张图，ESC退出
                break


if __name__ == "__main__":
    display('./release/mnist_lenet.pb')