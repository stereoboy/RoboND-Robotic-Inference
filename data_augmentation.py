import cv2
import numpy as np
import os

IMAGE_SIZE = 256
AUG_DIR = "./data/MixedNuts_aug"
TOTAL_AUG_NUM = 18

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()*0.7
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocess_img(img):
    resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    margin = (IMAGE_SIZE)*0.2
    final_img = cv2.copyMakeBorder(resized_img, margin, margin, margin, margin, cv2.BORDER_REFLECT_101)
    #final_img = cv2.copyMakeBorder(resized_img, margin, margin, margin, margin, cv2.BORDER_CONSTANT)
    return final_img

def scale_rot_trans_image(image, scale_range=0.3, rot_range=60):
    # Translation
    scale = 1 + np.random.uniform()*scale_range
    new_size = int(scale*IMAGE_SIZE)
    image = cv2.resize(image, (new_size, new_size))

    margin = 100
    image = cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_REFLECT_101)

    # Rotation
    rot = (np.random.uniform() - 0.5)*rot_range
    cx, cy = (margin + new_size)/2, (margin + new_size)/2
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1)
    image = cv2.warpAffine(image,M,(margin + new_size, margin + new_size))

    # Crop
    x = margin + int((new_size - IMAGE_SIZE)*np.random.uniform())
    y = margin + int((new_size - IMAGE_SIZE)*np.random.uniform())
    image = image[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE,:]
    return image

def main():
    for dirpath, dirs, files in os.walk("./raw_data/MixedNuts"):
        for directory in dirs:
            path = os.path.join(AUG_DIR, directory)
            if not os.path.exists(path):
                os.makedirs(path)
        print(dirs)
        for f in files:
            img_file = os.path.join(dirpath, f)
            img = cv2.imread(img_file)
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            for idx in range(TOTAL_AUG_NUM):
                aug = augment_brightness_camera_images(img)
                aug = scale_rot_trans_image(aug)
                if idx >= TOTAL_AUG_NUM/3 and idx < TOTAL_AUG_NUM*2/3:
                    aug = cv2.flip(aug, 1)
                if idx >= TOTAL_AUG_NUM*2/3 and idx < TOTAL_AUG_NUM:
                    aug = cv2.flip(aug, 0)
                aug_f = os.path.splitext(f)[0] + "_" + str(idx) + os.path.splitext(f)[1]
                aug_path = os.path.join(os.path.join(AUG_DIR, os.path.basename(dirpath)), aug_f)
                print ("save to", aug_path)
                print(cv2.imwrite(aug_path, aug))
#                cv2.imshow("display", np.vstack([resized, aug]))
#                if cv2.waitKey(1000) & 0xFF == ord('q'):
#                    break



if __name__ == '__main__':
    main()
