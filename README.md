# Learn-Mnist-
## 声明
1. 这是我学习使用神经网络实现手写数字识别的练习；
2. 程序中network.py和mnist_loader.py使用了学习资料的源码，在此基础上修改；
3. 源码https://github.com/unexploredtest/neural-networks-and-deep-learning/tree/master
4. 作者的程序包括了CNN，向量机等更优势的模型，这里只是对最初级的神经网络模型进行了练习
## 文件说明
1. main是使用主文件
2. mnist_loader 是作者源码 加载mnist的数据
3. network 也是作者的源码，创建网络，并使用随机梯度下降SGD训练
4. network_cuda是我是同pytorch（而非像作者使用numpy）的修改，以便使用cuda进行GPU训练，但实际效果不尽人意
5. network_save_use增加了模型的save和load的“方法”
6. network_use包含了最基础图像处理，用于load训练好的模型和测试的图片，进行手写数字识别
## 使用方法
1. 基础用法：训练模型
   main文件第一部分
3. 测试用法：训练模型并进行手写数字识别
   main文件第二部分
