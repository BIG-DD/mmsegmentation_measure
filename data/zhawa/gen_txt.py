

import os
import shutil
import random
path_name='./jpg'
counts=os.listdir(path_name)
with open('train1.txt','w') as file_object:
    for count in range(0,700):#进入到文件夹内，对每个文件进行循环遍历

        # os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
        if os.path.join(path_name,counts[count]).endswith('.jpg'):
            # print( os.path.join(path_name,counts[count]))
            str=os.path.basename( os.path.join(path_name,counts[count]))
            # print(str)
            x = str.split(".", 1)
            # print(x)
            file_numper=x[0]
            file_object.write(file_numper)
            file_object.write('\n')

with open('val1.txt','w') as file_object:
    for count in range(700,792):#进入到文件夹内，对每个文件进行循环遍历

        # os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
        if os.path.join(path_name,counts[count]).endswith('.jpg'):
            # print( os.path.join(path_name,counts[count]))
            str=os.path.basename( os.path.join(path_name,counts[count]))
            # print(str)
            x = str.split(".", 1)
            # print(x)
            file_numper=x[0]
            file_object.write(file_numper)
            file_object.write('\n')

with open('test1.txt','w') as file_object:
    for count in range(0,len(counts)):#进入到文件夹内，对每个文件进行循环遍历
        
        
        if os.path.join(path_name,counts[count]).endswith('.jpg'):
            print( os.path.join(path_name,counts[count]))
            str=os.path.basename( os.path.join(path_name,counts[count]))
            print(str)
            x = str.split(".", 1)
            print(x)
            file_numper=x[0]
            file_object.write(file_numper)
            file_object.write('\n')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
print(optimizer)