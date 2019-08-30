from __future__ import division
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import pprint

if __name__ == '__main__':
    hdf_name = 'datas/MCD43A3C.A2019227.h27v05.006.2019236003508.hdf'
    hdf_obj = SD(hdf_name)
    print(hdf_obj.info())
    data_dic = hdf_obj.datasets()
    for idx,sds in enumerate(data_dic.keys()):
        print(idx,sds)
    bsa = hdf_obj.select('Albedo_WSA_Band1')
    data = bsa.get()
    data = np.transpose(data)
    uee = [
        [82, 18, 25, 32, 25, 14, 25],
        [85, 18, 25, 32, 25, 14, 25],
        [90, 18, 25, 32, 25, 14, 25],
        [100, 18, 25, 32, 25, 14, 25],
        [125, 18, 25, 32, 0, 14, 25],
        [150, 18, 25, 32, 25, 14, 25],
        [255, 18, 25, 32, 25, 14, 25]
    ]
    f_data = np.asfarray(data*255/32767)
    print(f_data)
    plt.imshow(f_data, cmap='Blues')
    plt.show()



