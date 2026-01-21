# 声明：学习MNIST手写识别的练习，网络训练，数据加载来源于以下链接
# 代码学习来源：https://github.com/unexploredtest/neural-networks-and-deep-learning

## ====================================1. 加载数据，训练模型，打印训练过程====================================
'''
# 调用文件：(1) main.py (2) mnist_loader.py  (3)network.py
#加载数据
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#配置网络:教材（github程序）做法——使用Numpy库
import network
net=network.Network([784,30,10])

# #配置网络: 用torch库，引入cuda加快训练
# import network_cuda
# net=network_cuda.Network([784,30,10])

#SGD算法迭代训练
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
'''


## ====================================2. 加载数据，训练模型，保存模型，可视化模型训练过程====================================
# 调用文件：(1) main.py (2) mnist_loader.py  (3)network_save_use.py  (4)model_use.py
# #模型训练
# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#
# import network_save_use
# net=network_save_use.Network([784,30,10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#
# # 保存模型
# model_path = r"D:\bit_data\item\MNIST手写数字识别\my_mnist\model\mnist_model.pkl"
# net.save(model_path)
# print("模型已保存：", model_path)

#测试使用
import model_use
#加载训练好的模型
model_path = r"D:\bit_data\item\MNIST手写数字识别\my_mnist\model\mnist_model.pkl"
#加载要识别的图片
img_path = r"D:\bit_data\item\MNIST手写数字识别\my_mnist\test_pic\1.png"
#预测
pred, probs = model_use.predict_image(
    model_path=model_path,
    img_path=img_path,
    invert=True,
    binarize=False,
    topk=10
)