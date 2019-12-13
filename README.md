## mnist手写数字识别

### 训练过程
#### 1 数据准备
http://yann.lecun.com/exdb/mnist/  
从上面的网址下载mnist数据，mnist数据包括4个文件，如下所示  
t10k-images-idx3-ubyte.gz  
train-images-idx3-ubyte.gz  
t10k-labels-idx1-ubyte.gz  
train-labels-idx1-ubyte.gz  
#### 2 生成tfrecord
运行make_tfrecord.py生成tfrecord格式的数据，请将代码中的 '/home/datasets/mnist' 替换成你的数据路径  
运行结束之后会在make_tfrecord.py的同级目录生成一个tfrecord目录，其中包含两个tfrecord数据文件  
#### 3 训练
运行lenet.py进行训练，训练结束后终端显示类似下面的信息     
iter 1798  train accuracy 1.000000  loss 0.274144  test accuracy 0.998000  
iter 1799  train accuracy 1.000000  loss 0.167662  test accuracy 0.998000  
iter 1800  train accuracy 0.998000  loss 3.949928  test accuracy 0.990000   
训练过程中出现的ckpt目录存储了网络的参数，logs中存储了训练日志，可以利用tensorboard来查看    
#### 4 封装模型  
运行freeze_graph.py将ckpt中的网络参数提取出来，固化成pb格式的模型  
pb模型在./release目录中  
#### 5 测试模型准确度
运行model_test.py计算模型在全量测试集上的准确度，代码运行完成之后输出类似于下面的信息  
全量mnist测试集上的准确度为 0.9914   


### 代码说明
#### 1 release_mnist.py  
解析mnist数据，可以将图片写入存储，也可以返回ndarray
#### 2 make_tfrecord.py
将mnist数据转成tfrecord格式
#### 3 LeNet.py
读取tfrecord数据，训练网络
#### 4 freeze_graph.py
将代码封装成pb格式，便于传输和调用
#### 5 model_test.py  
测试封装的pb模型在测试集上的准确度
#### 6 display.py  
模型预测可视化，同时显示图像和预测结果      