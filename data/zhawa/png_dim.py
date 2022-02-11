import cv2
import os
import numpy as np
from PIL import Image

png_list = os.listdir('D:/liuxz_code/python/mmsegmentation/data/sashaguan/mask1')
for png_l in png_list:
    image = Image.open('D:/liuxz_code/python/mmsegmentation/data/sashaguan/mask1/' + png_l).convert('P')
    image.save('D:/liuxz_code/python/mmsegmentation/data/sashaguan/mask/' + png_l)