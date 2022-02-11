import os
import random

path_name = './jpg'
countss = os.listdir(path_name)
counts = random.sample(countss, len(countss))
with open('train.txt', 'w') as file_object:
    for count in range(0, len(counts)-100):  # 进入到文件夹内，对每个文件进行循环遍历
        if os.path.join(path_name, counts[count]).endswith('.jpg'):
            str = os.path.basename( os.path.join(path_name ,counts[count]))
            x = str.split(".", 1)
            file_numper = x[0]
            file_object.write(file_numper)
            file_object.write('\n')

with open('val.txt' ,'w') as file_object:
    for count in range(len(counts)-100, len(counts)-20):  # 进入到文件夹内，对每个文件进行循环遍历

        # os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
        if os.path.join(path_name ,counts[count]).endswith('.jpg'):
            # print( os.path.join(path_name,counts[count]))
            str =os.path.basename( os.path.join(path_name ,counts[count]))
            # print(str)
            x = str.split(".", 1)
            # print(x)
            file_numper =x[0]
            file_object.write(file_numper)
            file_object.write('\n')

with open('test.txt' ,'w') as file_object:
    for count in range(len(counts)-20, len(counts)):  # 进入到文件夹内，对每个文件进行循环遍历


        if os.path.join(path_name ,counts[count]).endswith('.jpg'):
            print( os.path.join(path_name ,counts[count]))
            str =os.path.basename( os.path.join(path_name ,counts[count]))
            print(str)
            x = str.split(".", 1)
            print(x)
            file_numper =x[0]
            file_object.write(file_numper)
            file_object.write('\n')