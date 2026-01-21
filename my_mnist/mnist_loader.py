"""
mnist_loader
~~~~~~~~~~~~

用于加载 MNIST 手写数字数据集的工具库。

关于返回数据结构的细节，请查看 ``load_data`` 和 ``load_data_wrapper``
两个函数的文档字符串说明。实际使用中，我们的神经网络代码通常调用
``load_data_wrapper``，因为它返回的数据格式更适合训练与测试。
"""

#### 依赖库
# 标准库
import pickle
import gzip

# 第三方库
import numpy as np


def load_data():
    """
    以元组形式返回 MNIST 数据：
    (training_data, validation_data, test_data)

    - training_data：训练集
    - validation_data：验证集
    - test_data：测试集

    其中 training_data 是一个二元组 (images, labels)：
    1) images：numpy.ndarray，共 50,000 条样本
       每条样本是一个长度为 784 的 numpy.ndarray，
       对应 28×28=784 个像素（通常是已展平的向量）。
    2) labels：numpy.ndarray，共 50,000 个数字标签，
       每个标签为 0~9 的整数，分别对应 images 中的图像。

    validation_data 与 test_data 的结构类似，
    但每个仅包含 10,000 张图像及对应标签。

    该数据格式本身很清晰，但在神经网络训练中通常希望：
    - x 为形状 (784, 1) 的列向量；
    - y 为 one-hot（10维单位向量）形式的标签；
    因此我们提供了 ``load_data_wrapper()`` 用于进一步转换数据格式。
    """
    # 从压缩文件读取数据（注意路径：../data/mnist.pkl.gz）
    f = gzip.open('D:\\bit_data\item\MNIST手写数字识别\my_mnist\data\mnist.pkl.gz', 'rb')

    # 兼容 Python2/3 的 pickle 读取方式：指定 latin1 编码
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'

    training_data, validation_data, test_data = u.load()
    f.close()

    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    返回 (training_data, validation_data, test_data)，
    在 ``load_data`` 的基础上，将数据转换为更适合神经网络实现的格式。

    1) training_data：
       一个长度为 50,000 的列表，每个元素是 (x, y) 二元组：
       - x：形状为 (784, 1) 的 numpy.ndarray（输入图像列向量）
       - y：形状为 (10, 1) 的 numpy.ndarray（对应标签的 one-hot 单位向量）

    2) validation_data 与 test_data：
       都是长度为 10,000 的列表，每个元素是 (x, y) 二元组：
       - x：形状为 (784, 1) 的 numpy.ndarray（输入图像列向量）
       - y：整数标签（0~9），即分类结果

    注意：训练集与验证/测试集的标签格式不同：
    - 训练集使用 one-hot，便于计算输出层误差与梯度；
    - 验证/测试集使用整数标签，便于评估正确率。
    这种“混合格式”在本网络代码中使用起来最方便。
    """
    tr_d, va_d, te_d = load_data()

    # 训练集输入：将每个 784 向量 reshape 成 (784, 1) 列向量
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]

    # 训练集标签：将 0~9 的整数标签转为 10 维 one-hot 列向量
    training_results = [vectorized_result(y) for y in tr_d[1]]

    # 组合成 (x, y) 列表
    training_data = list(zip(training_inputs, training_results))

    # 验证集：输入 reshape 为 (784, 1)，标签保持整数形式
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # 测试集：输入 reshape 为 (784, 1)，标签保持整数形式
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    返回一个 10 维单位向量（one-hot），第 j 个位置为 1.0，其余为 0.0。
    用于把数字标签 (0~9) 转成神经网络期望输出形式。

    例如 j=3，则返回：
    [0,0,0,1,0,0,0,0,0,0]^T
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
