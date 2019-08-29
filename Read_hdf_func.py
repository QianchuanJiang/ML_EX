from __future__ import division
from pyhdf.SD import SD, SDC
import numpy as np
import pprint

if __name__ == '__main__':
    hdf_name = 'datas/MCD43A3C.A2019227.h27v05.006.2019236003508.hdf'
    hdf_obj = SD(hdf_name)
    print(hdf_obj.info())
    data_dic = hdf_obj.datasets()
    for idx,sds in enumerate(data_dic.keys()):
        print(idx,sds)
    bsa = hdf_obj.select('Albedo_BSA_vis')
    data = bsa.get()
    data = np.transpose(data)
    print(data)
    print(data.shape)
    # ds



