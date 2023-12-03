import sys
import numpy as np
str1= sys.argv[1]
str2= sys.argv[2]
bias = [0.8, 0.85, 0.9, 0.95]

for b in bias:
    fname1 = str1 + str(b) + str2 + "31.txt"
    with open(fname1, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        last_line = lines[-1]
        result1=last_line[:4]

    fname2 = str1 + str(b) + str2 + "32.txt"
    with open(fname2, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        last_line = lines[-1]
        result2=last_line[:4]

    fname3 = str1 + str(b) + str2 + "33.txt"
    with open(fname3, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        last_line = lines[-1]
        result3=last_line[:4]

    fname4 = str1 + str(b) + str2 + "34.txt"
    with open(fname4, 'r') as f:  #打开文件
        lines = f.readlines() #读取所有行
        last_line = lines[-1]
        result4=last_line[:4]
    result = np.array([float(result1), float(result2), float(result3), float(result4)])
    print("bias:", b)
    print("mean:", np.mean(result))
    print("std:", np.std(result))
