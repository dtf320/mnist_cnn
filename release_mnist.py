# encoding:utf-8

import numpy as np
from struct import unpack
import gzip
import cv2
import os


# 将mnist数据集解压并读取图片存在train和test目录中


class MNIST:
    TRAIN_DATA_FILE = "train-images-idx3-ubyte.gz"
    TRAIN_LABEL_FILE = "train-labels-idx1-ubyte.gz"
    TEST_DATA_FILE = "t10k-images-idx3-ubyte.gz"
    TEST_LABEL_FILE = "t10k-labels-idx1-ubyte.gz"

    def read_image(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))  # > 表示大端模式, 4I表示4个unsigned int,每个unsigned int占4个字节
            fmt = str(num*rows*cols)+"B"                        # B代表unsigned char, 对应python中的integer, 占1个字节
            image = unpack(fmt,f.read())
            image = np.asarray(image,dtype=np.uint8).reshape([num,28,28,1])
        return image

    def read_label(self, path):
        with gzip.open(path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            fmt = str(num)+"B"                        # B代表unsigned char, 对应python中的integer, 占1个字节
            d = unpack(fmt,f.read())
        return d

    def load(self, data_path):
        train_image = self.read_image(os.path.join(data_path,self.TRAIN_DATA_FILE))
        test_image = self.read_image(os.path.join(data_path,self.TEST_DATA_FILE))
        train_label = self.read_label(os.path.join(data_path,self.TRAIN_LABEL_FILE))
        test_label = self.read_label(os.path.join(data_path,self.TEST_LABEL_FILE))
        return train_image,test_image,train_label,test_label

    def mat2bmp(self, idx, mat, label, train_or_test, release_path):
        im_path = os.path.join(release_path,train_or_test)
        if not os.path.exists(im_path):
            os.makedirs(im_path)
        cv2.imwrite(os.path.join(im_path,"train-"+str(idx).rjust(5,"0")+"-"+str(label)+".bmp"),mat)


if __name__ == "__main__":
    mnist = MNIST()
    train_image,test_image,train_label,test_label = mnist.load('/home/datasets/mnist')

    for idx,img in enumerate(train_image):
        mnist.mat2bmp(idx,img.reshape(28,28),train_label[idx],"train","./data")

    for idx,img in enumerate(test_image):
        mnist.mat2bmp(idx,img.reshape(28,28),test_label[idx],"test","./data")
