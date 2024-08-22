from qplib import QPLIB
import pandas as pd
import json 

folder_path = './qplib_ori/'

# 连续变量的18个问题
ids = ['10034','8495','8500','8547','8559','8567','8602','8616','8785',\
       '8790','8792','8845','8906','8938','8991','9002','9008']

# 整数变量的14个问题
ids = ['10050','10056','3547','3694','3698','3708','3792','3861','3871',\
       '3913','5527','5543','5577','5924']
ids = ['3547', '3694', '3698', '3708', '3792', '3861', '3871', '5527', '5543', '5577', '5924']
# 计算特征值有问题的
# ids = ['10038','4270',]
# 非凸的
# ids = ['3980','10069']

#ids = ['10050']

for id in ids:
       print(id)
       name = 'QPLIB_' + id
       path = folder_path + name + '.qplib'
       #    instance = QPLIB(path,True)
       instance = QPLIB(path)

       instance.copt_convert(name,solve=False)
       instance
