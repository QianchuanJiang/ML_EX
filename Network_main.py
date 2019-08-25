import numpy as np
# 导入sigmoid函数；
import scipy.special

# BP神经网络基础类；
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入层，隐藏层，输出层
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate
        # 权重；统一使用正态分布的方式生成, 标准方差设置为节点传入链接数目的开方；
        # 输入层到隐藏层；
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 隐藏层到输出层；
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 激活函数（使用sigmoid函数来当做激活函数）;
        self.activation_function = lambda x: scipy.special.expit(x)


    # 训练方法；
    def train(self, inputs_list, targets_list):
        # 输入层；
        inputs = np.array(inputs_list, ndmin=2).T
        # 导入目标值数据集；
        targets = np.array(targets_list, ndmin=2).T
        # 输入层到隐藏层(公式：X hidden = W * I)；
        hidden_inputs = np.dot(self.wih, inputs)
        # 引入到激活函数sigmiod中；
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层到输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算出误差值，用于反向传播；
        output_errors = targets - final_outputs
        # 根据所连接的权重分割误差，为每个隐藏节点重组这些误差，
        # 公式为：ERRPRS(hidden) = WEIGHTS(hidden_output) • ERRORS(output)
        hidden_errors = np.dot(self.who.T, output_errors)
        # 更新隐藏层和输出层之前权重；
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))


    # 查询方法；
    def query(self, inputs_lsit):
        # 把输入的数组转换成二维数组；
        inputs = np.array(inputs_lsit, ndmin=2).T
        # 计算输入层到隐藏层之间的权重矩阵乘以输入矩阵；
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算输出层,并加入激活函数；
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层到输出层,同样加入激活函数；
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outpus = self.activation_function(final_inputs)
        return final_outpus

    # 主运行方法；
    # dasdfa
    if __name__ == '__main__':
        pass
    


