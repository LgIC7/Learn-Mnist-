# model_use.py
# -----------------------------------------
# 用于加载已训练好的 MNIST 模型，并对单张图片进行数字识别
# 输出 0~9 的“可能性大小”（这里用输出层激活值经 softmax 归一化为概率）

import numpy as np
from PIL import Image
from network_save_use import Network


def softmax(z):
    """
    softmax：把10维输出归一化成概率
    z: (10,1)
    return: (10,1)
    """
    z = z - np.max(z)  # 数值稳定
    expz = np.exp(z)
    return expz / np.sum(expz)


def load_model(model_path):
    """加载模型，返回 Network 对象"""
    net = Network.load(model_path)
    return net


def preprocess_image(img_path, invert=True, binarize=False):
    """
    读取图片并预处理成 MNIST 输入格式：(784,1)，像素范围 [0,1]

    img_path: 图片路径（任意尺寸均可）
    invert: 是否反色（MNIST：黑底白字；你的图片若是白底黑字通常需要反色）
    binarize: 是否二值化（一般不建议，先关掉；若图片噪声大可尝试）

    返回：
      x: (784,1) numpy.ndarray, float64, 范围[0,1]
    """
    img = Image.open(img_path).convert("L")   # 灰度图
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # 缩放到28x28

    arr = np.array(img).astype(np.float64)    # (28,28), 0~255

    # 是否反色：白底黑字 → 反色后更接近MNIST的“黑底白字”
    if invert:
        arr = 255.0 - arr

    # 归一化到 0~1
    arr = arr / 255.0

    # 可选二值化（阈值可调）
    if binarize:
        arr = (arr > 0.5).astype(np.float64)

    # 展平为 (784,1)
    x = arr.reshape(784, 1)
    return x


def predict_with_probs(net, x):
    """
    输入 x (784,1)，输出：
    - pred: 预测类别 0~9
    - probs: 10维概率（softmax后），shape (10,)
    """
    out = net.feedforward(x)          # out: (10,1)，sigmoid输出
    probs = softmax(out).reshape(-1)  # (10,)
    pred = int(np.argmax(probs))
    return pred, probs


def predict_image(model_path, img_path, invert=True, binarize=False, topk=10):
    """
    一步到位：加载模型 + 读图 + 输出预测与概率

    返回：
      pred: int
      probs: np.ndarray (10,)
    """
    net = load_model(model_path)
    x = preprocess_image(img_path, invert=invert, binarize=binarize)
    pred, probs = predict_with_probs(net, x)

    # 按概率从大到小排序输出
    order = np.argsort(-probs)
    print("图片：", img_path)
    print("预测结果：", pred)
    print("0~9 概率（softmax归一化）：")
    for i in order[:topk]:
        print("  {} : {:.6f}".format(i, probs[i]))

    return pred, probs
