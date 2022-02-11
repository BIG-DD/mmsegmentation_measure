import cv2
import os

png_list = os.listdir('D:/liuxz_code/python/mmsegmentation/data/sashaguan/png')
for png_i in png_list:
    png_joj = cv2.imread('D:/liuxz_code/python/mmsegmentation/data/sashaguan/png/' + png_i)
    x = png_i.split('.')
    cv2.imwrite('D:/liuxz_code/python/mmsegmentation/data/sashaguan/jpg/{}.jpg'.format(x[0]), png_joj)