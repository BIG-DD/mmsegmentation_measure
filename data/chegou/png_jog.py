import cv2
import os

png_list = os.listdir('jpj')
for png_i in png_list:
    png_joj = cv2.imread('jpj/' + png_i)
    x = png_i.split('.')
    cv2.imwrite('jpg/{}.jpg'.format(x[0]), png_joj)