import os
import cv2
import torch.utils.data as data
import random
import numpy as np


def BGR2HSV(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

def HSV2BGR(img):
    return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
def randomFlip(imgs):   
    for i in range(0,len(imgs)):
        # if random .random()<0.5:
        raw_img=cv2.imread(os.path.join(path_raw,imgs[i][0]))
        mask_img=cv2.imread(os.path.join(path_mask,imgs[i][1]))
        bgr_raw = np.fliplr(raw_img).copy()
        bgr_mask = np.fliplr(mask_img).copy()
        save_raw_name=os.path.join(save_raw,("random_flip_"+imgs[i][0]))
        save_mask_img=os.path.join(save_mask,("random_flip_"+imgs[i][1]))
        cv2.imwrite(save_raw_name,bgr_raw)
        cv2.imwrite(save_mask_img,bgr_mask)

def randomBrightness(imgs):
    for i in range(0,len(imgs)):
        # if random .random()<0.5:
        raw_img=cv2.imread(os.path.join(path_raw,imgs[i][0])).copy()
        mask_img=cv2.imread(os.path.join(path_mask,imgs[i][1])).copy()
        
        bgr_raw =BGR2HSV(raw_img) 
        h,s,v=cv2.split(bgr_raw)
        adjust=random.choice([0.5,1.5])
        v=v*adjust
        v=np.clip(v,0,255).astype(bgr_raw.dtype)
        bgr_raw=cv2.merge((h,s,v))
        save_raw_name=os.path.join(save_raw,("randomBrightness_"+imgs[i][0]))
        cv2.imwrite(save_raw_name,HSV2BGR(bgr_raw))
        save_mask_img=os.path.join(save_mask,("randomBrightness_"+imgs[i][1]))
        cv2.imwrite(save_mask_img,mask_img)

def randomScale(imgs):
    for i in range(0,len(imgs)):
        # if random .random()<0.5:
        scale=random.uniform(0.8,1.2)
        raw_img=cv2.imread(os.path.join(path_raw,imgs[i][0])).copy()
        mask_img=cv2.imread(os.path.join(path_mask,imgs[i][1])).copy()
        
        height_raw, width_raw, c_raw = raw_img.shape
        raw_img = cv2.resize(raw_img, (int(width_raw * scale), height_raw))
        
        height_mask, width_mask, c_mask = mask_img.shape
        mask_img = cv2.resize(mask_img, (int(width_mask * scale), height_mask))
        
        save_raw_name=os.path.join(save_raw,("randomScale_"+imgs[i][0]))
        cv2.imwrite(save_raw_name,raw_img)
        save_mask_img=os.path.join(save_mask,("randomScale_"+imgs[i][1]))
        cv2.imwrite(save_mask_img,mask_img)

def randomSaturation(imgs):
    for i in range(0,len(imgs)):
        # if random.random()<0.5:
        raw_img = cv2.imread(os.path.join(path_raw, imgs[i][0])).copy()
        mask_img = cv2.imread(os.path.join(path_mask, imgs[i][1])).copy()

        hsv_raw = BGR2HSV(raw_img)
        h, s, v = cv2.split(hsv_raw)
        adjust = random.choice([0.5, 1.5])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv_raw.dtype)
        hsv_raw = cv2.merge((h, s, v))

        save_raw_name = os.path.join(save_raw, ("randomSaturation_" + imgs[i][0]))
        cv2.imwrite(save_raw_name, HSV2BGR(hsv_raw))
        save_mask_img = os.path.join(save_mask, ("randomSaturation_" + imgs[i][1]))
        cv2.imwrite(save_mask_img, mask_img)


def randomHue(imgs):
    for i in range(0, len(imgs)):
        # if random.random() < 0.5:
        raw_img = cv2.imread(os.path.join(path_raw, imgs[i][0])).copy()
        mask_img = cv2.imread(os.path.join(path_mask, imgs[i][1])).copy()
        hsv_raw = BGR2HSV(raw_img)
        h, s, v = cv2.split(hsv_raw)
        adjust = random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv_raw.dtype)
        hsv_raw = cv2.merge((h, s, v))
        save_raw_name = os.path.join(save_raw, ("randomHue_" + imgs[i][0]))
        cv2.imwrite(save_raw_name, HSV2BGR(hsv_raw))
        save_mask_img = os.path.join(save_mask, ("randomHue_" + imgs[i][1]))
        cv2.imwrite(save_mask_img, mask_img)

def rangdomBlur(imgs):
	for i in range(0,len(imgs)):
		raw_img = cv2.imread(os.path.join(path_raw, imgs[i][0])).copy()
		mask_img = cv2.imread(os.path.join(path_mask, imgs[i][1])).copy()
		save_raw_name = os.path.join(save_raw, ("rangdomBlure_" + imgs[i][0]))
		cv2.imwrite(save_raw_name, cv2.blur(raw_img,(5,5)))
		save_mask_img = os.path.join(save_mask, ("rangdomBlure_" + imgs[i][1]))
		cv2.imwrite(save_mask_img, mask_img)

    
if __name__=="__main__":
    
    path_raw=r"D:\liuxz_code\python\mmsegmentation\data\sashaguan\png"
    path_mask=r"D:\liuxz_code\python\mmsegmentation\data\sashaguan\mask"
    save_raw=r"D:\liuxz_code\python\mmsegmentation\data\sashaguan\encod_data\png"
    save_mask=r"D:\liuxz_code\python\mmsegmentation\data\sashaguan\encod_data\mask"
    raw_img_nums=os.listdir(path_raw)

    mask_img_nums=os.listdir(path_mask)
    imgs=[]

    if len(raw_img_nums)>len(mask_img_nums):
        for num in range(0,len(mask_img_nums)):
            
            if os.path.join(path_raw,raw_img_nums[num]).endswith('.png') and os.path.join(path_raw,mask_img_nums[num]).endswith('.png'):
                if str(raw_img_nums[num])==str(mask_img_nums[num]):
                    img=[]
                    img.append(raw_img_nums[num]) 
                    img.append(mask_img_nums[num])
                imgs.append(img)        
    else:
        for num in range(0,len(raw_img_nums)):
            img=[]
            if os.path.join(path_raw,raw_img_nums[num]).endswith('.png') and os.path.join(path_raw,mask_img_nums[num]).endswith('.png'):
                if str(raw_img_nums[num])==str(mask_img_nums[num]):
                    img.append(raw_img_nums[num])
                    img.append(mask_img_nums[num])
                imgs.append(img)  

    random.shuffle(imgs)

    randomFlip(imgs)
    randomBrightness(imgs)
    randomScale(imgs)
    randomHue(imgs)
    rangdomBlur(imgs)
    randomSaturation(imgs)

