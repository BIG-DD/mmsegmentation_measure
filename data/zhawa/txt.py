#实现数据增强：图像和label的镜像、旋转、加噪声、亮度调节

import cv2
import numpy as np
import random

def load_data(data_path):
    """
    读取图像数据
    :param data_path:图像数据路径
    :return: 图像数据
    """
    img = cv2.imread(data_path)
    return img

def load_xy(txt_path):
    """
    读取txt label文件中的点坐标值
    :param txt_path: txt label 坐标路径
    :return: 每个点的 x y 坐标
    """
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            tem = line.split()
            del tem[0]
            x_y = tem
    return x_y

def img_flit(img):
    """
    实现镜像翻转
    :param img:输入待翻转的图像
    :return: 翻转后的图像
    """
    img = cv2.flip(img, 1)
    return img

def txt_flit(img, x_y):
    """
    对txt标签文件进行镜像翻转
    :param img: 读取标签对应的图片，用于取宽度值
    :param x_y: txt标签的xy值
    :return: 镜像后的xy值
    """
    x_yf = []
    img_w = img.shape[1]
    xf_1 = int(img_w) - int(x_y[0])
    x_yf.append(xf_1)
    x_yf.append(x_y[1])
    xf_2 = int(img_w) - int(x_y[2])
    x_yf.append(xf_2)
    x_yf.append(x_y[3])
    xf_3 = int(img_w) - int(x_y[4])
    x_yf.append(xf_3)
    x_yf.append(x_y[5])
    xf_4 = int(img_w) - int(x_y[6])
    x_yf.append(xf_4)
    x_yf.append(x_y[7])
    return x_yf

def line_in_img(img, xy):
    """
    在图像上按照给定的点画出框来
    :param img: 输入图像
    :param xy: 画框的四个点
    :return: 画了框的图像
    """
    tl = round(0.001 * max(img.shape[0:2])) + 1  # line thickness
    color = [0, 0, 255]
    cv2.line(img, (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3])), color, tl)
    cv2.line(img, (int(xy[2]), int(xy[3])), (int(xy[4]), int(xy[5])), color, tl)
    cv2.line(img, (int(xy[4]), int(xy[5])), (int(xy[6]), int(xy[7])), color, tl)
    cv2.line(img, (int(xy[6]), int(xy[7])), (int(xy[0]), int(xy[1])), color, tl)
    return img

def show_img(img, name):
    """
    用于显示图像
    :param img:输入图像
    :param name: 图像显示框的名字
    :return:
    """
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, 400, 300)
    cv2.imshow(name, img)
    cv2.waitKey(0)

def add_noise(img, noise_name):
    """
    给图像增加噪声
    :param img:输入待增加噪声的图像
    :param noise_name: 增加噪声的名字,"gauss"表示高斯噪声，“salt_pepper”表示椒盐噪声
    :return: 加了噪声的图像
    """
    if noise_name == "gauss":
        img = img.astype(np.int16)  # 此步是为了避免像素点小于0，大于255的情况
        mu = 0
        sigma = 10
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        return img

    elif noise_name == "salt_pepper":
        salt = np.zeros(img.shape, np.uint8)
        prob = 0.001
        thres = 1 - prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = np.random.rand()
                if rdn < prob:
                    salt[i][j] = 0
                elif rdn > thres:
                    salt[i][j] = 255
                else:
                    salt[i][j] = img[i][j]
        return salt
    else:
        print("输入正确的噪声名")

def img4to_1(base_path,img_name):
    """
    将四张图像拼接在一起，要求拼接前的图像尺寸一致，拼接后的图像尺寸与原图尺寸一致
    :param base_path: 图像存放基本路径
    :param img_name: fan
    :return:
    """
    img1 = cv2.imread('dataset/3000.png')
    img2 = cv2.imread('dataset/10.png')
    img3 = cv2.imread('dataset/180.png')
    img4 = cv2.imread('dataset/100.png')
    img_h = img1.shape[0]
    img_w = img1.shape[1]
    img = np.zeros([img_h, img_w, 3], dtype=np.uint8)
    img.fill(255)
    img1 = cv2.resize(img1, (int(img_w / 2), int(img_h / 2)), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (int(img_w / 2), int(img_h / 2)), interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img3, (int(img_w / 2), int(img_h / 2)), interpolation=cv2.INTER_AREA)
    img4 = cv2.resize(img4, (int(img_w / 2), int(img_h / 2)), interpolation=cv2.INTER_AREA)
    img[0:int(img_h / 2), 0:int(img_w / 2)] = img1
    img[0:int(img_h / 2), int(img_w / 2):int(img_w)] = img2
    img[int(img_h / 2):int(img_h), 0:int(img_w / 2)] = img3
    img[int(img_h / 2):int(img_h), int(img_w / 2):int(img_w)] = img4
    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', 400, 300)
    cv2.imshow('img', img)
    cv2.waitKey(0)
if __name__ == '__main__':

    imgname = './dataset/3000.png'
    label_txt = './dataset/3000.txt'
    img_y = load_data(imgname)
    img_yl = add_noise(img_y, 'salt_pepper')
    show_img(img_y, "yyy")
    show_img(img_yl, 'ccc')
