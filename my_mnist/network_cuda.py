"""
network.py
~~~~~~~~~~

本模块实现前馈神经网络的随机梯度下降（SGD）训练，
梯度由反向传播（backpropagation）计算。

说明：代码强调简单易读，未做性能优化。
"""

# 标准库
import random
import time

# 第三方库：用 torch 替代 numpy，从而支持 .cuda()
import torch


class Network(object):

    def __init__(self, sizes, use_cuda=True):
        """
        sizes：各层神经元数量，例如 [784, 30, 10]
        use_cuda：是否使用 GPU（若 cuda 不可用则自动回退 CPU）
        """
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 参数初始化：高斯 N(0,1)，并放到 device 上
        self.biases = [torch.randn(y, 1, device=self.device) for y in sizes[1:]]
        self.weights = [torch.randn(y, x, device=self.device)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """前向传播：输入 a，返回网络输出"""
        for b, w in zip(self.biases, self.weights):
            a = torch.sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        使用 mini-batch SGD 训练。
        training_data: [(x, y_onehot), ...]
        test_data: [(x, y_int), ...]
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            t1 = time.time()
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                # ======= 关键：训练环节把数据放到 GPU（.cuda） =======
                if self.use_cuda:
                    mini_batch = [
                        (torch.as_tensor(x, dtype=torch.float32).cuda(),
                         torch.as_tensor(y, dtype=torch.float32).cuda())
                        for (x, y) in mini_batch
                    ]
                else:
                    mini_batch = [
                        (torch.as_tensor(x, dtype=torch.float32),
                         torch.as_tensor(y, dtype=torch.float32))
                        for (x, y) in mini_batch
                    ]
                # ======================================================
                self.update_mini_batch(mini_batch, eta)

            t2 = time.time()

            if test_data:
                acc = self.evaluate(test_data)  # evaluate 内部也会 .cuda()
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    j, acc, n_test, t2 - t1
                ))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(j, t2 - t1))

    def update_mini_batch(self, mini_batch, eta):
        """用一个 mini-batch 的平均梯度更新参数"""
        nabla_b = [torch.zeros_like(b) for b in self.biases]
        nabla_w = [torch.zeros_like(w) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """对单样本 (x,y) 计算梯度，返回 (nabla_b, nabla_w)"""
        nabla_b = [torch.zeros_like(b) for b in self.biases]
        nabla_w = [torch.zeros_like(w) for w in self.weights]

        # ---------- 前向传播：保存每层 z 和 activation ----------
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = w @ activation + b
            zs.append(z)
            activation = torch.sigmoid(z)
            activations.append(activation)

        # ---------- 反向传播 ----------
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta @ activations[-2].t()

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = self.weights[-l + 1].t() @ delta * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta @ activations[-l - 1].t()

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        返回测试集中预测正确的样本数（y 为整数标签）
        """
        correct = 0
        for (x, y) in test_data:
            # ======= 关键：测试环节把 x 放到 GPU（.cuda） =======
            if self.use_cuda:
                x = torch.as_tensor(x, dtype=torch.float32).cuda()
            else:
                x = torch.as_tensor(x, dtype=torch.float32)
            # ======================================================
            out = self.feedforward(x)
            pred = int(torch.argmax(out).item())
            if pred == y:
                correct += 1
        return correct

    def cost_derivative(self, output_activations, y):
        """代价函数对输出层激活的偏导：∂C/∂a"""
        return (output_activations - y)

# torch 包含了sigmoid函数

def sigmoid_prime(z):
    """sigmoid 的导数：σ(z)(1-σ(z))"""
    s = torch.sigmoid(z)
    return s * (1.0 - s)
