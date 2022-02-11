import cv2
import numpy as np
import math

def get_4_point(img_path):
    """
    通过cv2中的函数，获得目标图像的最小外接矩形的四个点坐标及目标区域中所有像素点的坐标
    :param img_path: 语义分割后，只包含一个预测区域的图像
    :return: box1:[[x1, y1], [x2, y2]...]shape为（4，2）的数组，包含四个坐标值; aa:[[x1, y1], [x2, y2], ...]目标区域中所有像素点坐标
    """
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(img.shape)
    print(type(img[672, 672]))
    ret, thresh = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY_INV)
    print(type(img))
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:  # 遍历轮廓
        rect = cv2.minAreaRect(c)  # 生成最小外接矩形
        box_ = cv2.boxPoints(rect)
        box1 = np.int0(box_)
        print("box1:", box1)
        print("box1min", max(box1[:, 0]))
        #cv2.drawContours(img, [box1], -1, (255, 0, 0), 1)
        print('img.shape', img.shape)
        a = []
        for j in range(min(box1[:, 0]), max(box1[:, 0]) + 1):
            for i in range(min(box1[:, 1]), max(box1[:, 1]) + 1):
                #print("img(i, j)",img[i, j])
                if img[i, j] > 0:
                    a.append([j, i])
        aa = np.array(a)
        print("aa.shape",aa.shape)
    return aa, box1

def dist(point1, point2):
    """
    计算point1和point2的距离
    :param point11: [x1, y1]
    :param point22: [x2, y2]
    :return: 两点距离
    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    x1x2 = x1 - x2
    y1y2 = y1 - y2
    distt = math.sqrt(x1x2 ** 2 + y1y2 ** 2)
    return distt

def point_updown(point_4):
    """
    通过外接矩形的四个点算出两条长边的坐标，并返回两条长边的各三分点的坐标
    :param point_4:  [[x1, y1], [x2, y2]...]shape为（4，2）的数组，包含四个坐标值
    :return: point3_up, point3_down，分别包含两个的坐标
    """
    point1 = [point_4[0, 0], point_4[0, 1]]
    point2 = [point_4[1, 0], point_4[1, 1]]
    point3 = [point_4[2, 0], point_4[2, 1]]
    point4 = [point_4[3, 0], point_4[3, 1]]
    print(str(point1) + '\n' + str(point2) + '\n' + str(point3) + '\n' + str(point4))
    d_x1x2 = dist(point1, point2)
    d_x2x3 = dist(point2, point3)
    d_x3x4 = dist(point3, point4)
    d_x4x1 = dist(point4, point1)
    print("dis " + str(d_x1x2) + '\n' + str(d_x2x3) + '\n' + str(d_x3x4) + '\n' + str(d_x4x1))
    d_all = d_x1x2 + d_x2x3 + d_x3x4 +d_x4x1
    P1_x, P1_y = point1[0], point1[1]
    P2_x, P2_y = point2[0], point2[1]
    P3_x, P3_y = point3[0], point3[1]
    P4_x, P4_y = point4[0], point4[1]

    xy_up = []
    xy_down = []
    if d_x1x2 > (d_all / 4):
        x1_up = P2_x + ((P1_x - P2_x) / 4)
        x2_up = P2_x + ((P1_x - P2_x) / 2)
        x3_up = P2_x + ((3 * P1_x - 3 * P2_x) / 4)
        y1_up = (((x1_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
        y2_up = (((x2_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
        y3_up = (((x3_up - P1_x) * (P2_y - P1_y)) / (P2_x - P1_x)) + P1_y
        if (P1_x - P2_x) > 0:
            xy_up = [[x1_up, y1_up], [x2_up, y2_up], [x3_up, y3_up]]
        if (P1_x - P2_x) < 0:
            xy_up = [[x3_up, y3_up], [x2_up, y2_up], [x1_up, y1_up]]
    if d_x2x3 > (d_all / 4):
        x1_up = P3_x + ((P2_x - P3_x) / 4)
        x2_up = P3_x + ((P2_x - P3_x) / 2)
        x3_up = P3_x + ((3 * P2_x - 3 * P3_x) / 4)
        y1_up = (((x1_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
        y2_up = (((x2_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
        y3_up = (((x3_up - P2_x) * (P3_y - P2_y)) / (P3_x - P2_x)) + P2_y
        if (P2_x - P3_x) > 0:
            xy_up = [[x1_up, y1_up], [x2_up, y2_up], [x3_up, y3_up]]
        if (P2_x - P3_x) < 0:
            xy_up = [[x3_up, y3_up], [x2_up, y2_up], [x1_up, y1_up]]
    if d_x3x4 > (d_all / 4):
        x1_down = P4_x + ((P3_x - P4_x) / 4)
        x2_down = P4_x + ((P3_x - P4_x) / 2)
        x3_down = P4_x + ((3 * P3_x - 3 * P4_x) / 4)
        y1_down = (((x1_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
        y2_down = (((x2_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
        y3_down = (((x3_down - P3_x) * (P4_y - P3_y)) / (P4_x - P3_x)) + P3_y
        if (P3_x - P4_x) > 0:
            xy_down = [[x1_down, y1_down], [x2_down, y2_down], [x3_down, y3_down]]
        if (P3_x - P4_x) < 0:
            xy_down = [[x3_down, y3_down], [x2_down, y2_down], [x1_down, y1_down]]
    if d_x4x1 > (d_all / 4):
        x1_down = P1_x + ((P4_x - P1_x) / 4)
        x2_down = P1_x + ((P4_x - P1_x) / 2)
        x3_down = P1_x + ((3 * P4_x - 3 * P1_x) / 4)
        y1_down = (((x1_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
        y2_down = (((x2_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
        y3_down = (((x3_down - P4_x) * (P1_y - P4_y)) / (P1_x - P4_x)) + P4_y
        if (P4_x - P1_x) > 0:
            xy_down = [[x1_down, y1_down], [x2_down, y2_down], [x3_down, y3_down]]
        if (P4_x - P1_x) < 0:
            xy_down = [[x3_down, y3_down], [x2_down, y2_down], [x1_down, y1_down]]

    xyint3_up = np.round(np.array(xy_up))
    xyint3_down = np.round(np.array(xy_down))

    return xyint3_up, xyint3_down

def get_point2_incontourrs(point_up, point_down, point_contourrs, bata = 50.0):
    """
    给定目标轮廓外接矩形上对应的一对点，建立一条通过两点的直线。输出目标区域与直线首先接触的点和最后离开点的坐标。
    :param point_up: [x1, y1]外接矩形，其中一条长边上的某一点，本程序中是长边上的三分三分点
    :param point_down: [x1, y1]外接矩形，另一条长边上的对应点
    :param point_contourrs: [[x1, y1], [x2, y2],...]目标区域中所有点集的集合
    :return: [[x1, y1], [x2, y2]]目标区域上对应两点的坐标
    """
    x1, y1 = point_up[0], point_up[1]
    x2, y2 = point_down[0], point_down[1]
    numb= point_contourrs.shape[0]
    point_in_contourrs = np.array(point_contourrs)

    #经过给定的两个点，建立一个方程，(y - y1)(x2 - x1) - (x - x1)(y2 - y1) = mbx
    xishu = np.repeat(np.expand_dims(np.array([(y1 - y2), (x2 - x1)]), axis=0), numb, axis=0)
    cc = np.repeat(np.expand_dims(np.array([(x2 * y1 - x2 * y2), (y2 * x2 - y2 * x1)]), axis=0), numb, axis=0)
    ax = np.multiply(point_in_contourrs, xishu)
    bx = ax - cc
    mbx = bx.sum(axis=1)
    mbx = np.expand_dims(mbx, axis=1)
    hh, ll = mbx.shape
    point2_contourrs = []
    for hhh in range(hh):
        if mbx[hhh, 0] < bata:
            if mbx[hhh, 0] > -bata:
                #找出所建立直线附近点的坐标，把坐标放入point2_contourrs中
                pointss = point_in_contourrs[hhh, :]
                point2_contourrs.append(pointss)
    point2_contourrs = np.array(point2_contourrs)

    #把直线附近点的坐标x和y相加，用于找出其和的最大值和最小值
    pointssum = point2_contourrs.sum(axis=1)
    pointssum = np.expand_dims(pointssum, axis=1)
    hh_max, ll_max = np.where(pointssum == np.max(pointssum))
    hh_min, ll_min = np.where(pointssum == np.min(pointssum))
    pointss_max = point2_contourrs[hh_max[0], :]
    pointss_min = point2_contourrs[hh_min[0], :]

    #把点对应到img图像时注意x值对应图像W，y值对应图像H
    return pointss_max, pointss_min


if __name__ == '__main__':
    img_path = '../data/zhawa/png/00.png'
    point_in_contourrs, point_4 = get_4_point(img_path)
    # point_up, point_down = point_updown(point_4)
    # pointup_one = point_up[0, :]
    # pointdown_one = point_down[0, :]
    # point_one, point_two = get_point2_incontourrs(pointup_one, pointdown_one, point_in_contourrs)
    # print("one:", point_one, '  ', point_two)