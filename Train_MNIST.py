import numpy as np
import matplotlib.pyplot as plt
from Network_main import neuralNetwork

def read_datas(data_path):
    data_file = open(data_path)
    data_list = data_file.readlines()
    data_file.close()
    return data_list



if __name__ == '__main__':
    # 读取到训练集的每一行数据；
    data_list = read_datas('datas/mnist_train.csv')
    # # 获取第一张训练图片的全部数据；
    # all_values = data_list[100].split(',')
    # # 截取数据第一位以后的所有数据，并把他们转换成一个28*28的二维数组；
    # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # plt.show()

    # 设置网络，开始训练；
    input_nodes = 784
    hidden_modes = 100
    output_nodes = 10
    learning_rate = 0.1
    # dsfa

    # 实例化神经网络对象；
    n = neuralNetwork(inputnodes=input_nodes, hiddennodes=hidden_modes, outputnodes=output_nodes, learningrate=learning_rate)
    # 向网络内传入训练数据集；
    training_data_list = data_list
    for record in training_data_list:
        # 分割数据串；
        all_values = record.split(',')
        # 转换为训练特征集；通过np.asfarry方法把字符串类型的数据转换成实数，
        # 然后对这些数据进行归一化处理：把原有0到255之间的数据处理到0到1之间；
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 目标矩阵；类似于一个onehot类型的数据组，基础是0.01
        targers = np.zeros(output_nodes) + 0.01
        targers[int(all_values[0])] = 0.09
        # 把生成的训练特征集和目标集传入到神经网络对象的train方法中；
        n.train(inputs_list=inputs, targets_list=targers)

    # 测试结果
    pic_values = data_list[2300].split(',')
    # 截取数据第一位以后的所有数据，并把他们转换成一个28*28的二维数组；
    image_array = np.asfarray(pic_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    ex = n.query(np.asfarray(pic_values[1:]))
    print(ex)
    plt.show()



