import cv2
import os
import numpy as np
from PIL import Image

png_list = os.listdir('png_1')
for png_l in png_list:
    image = Image.open('png_1/' + png_l).convert('P')
    image.save('png/' + png_l)