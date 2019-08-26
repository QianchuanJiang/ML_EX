import numpy as np
import matplotlib.pyplot as plt

def read_datas(data_path):
    data_file = open(data_path)
    data_list = data_file.readlines()
    data_file.close()
    return data_list



if __name__ == '__main__':
    # 读取到训练集的每一行数据；
    data_list = read_datas('mnist_train.csv')
    # 获取第一张训练图片的全部数据；
    all_values = data_list[100].split(',')
    # 截取数据第一位以后的所有数据，并把他们转换成一个28*28的二维数组；
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()
    
