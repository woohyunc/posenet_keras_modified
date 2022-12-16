import cv2
import os

aug_directory = "../augmented_images/"
img_directory = "../images/"

def center_crop_vert(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = w
    crop_height = set_size

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img

def center_crop_horz(img, set_size):

    h, w, c = img.shape

    if set_size > min(h, w):
        return img

    crop_width = set_size
    crop_height = h

    mid_x, mid_y = w//2, h//2
    offset_x, offset_y = crop_width//2, crop_height//2
       
    crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
    return crop_img


for filename in os.listdir(img_directory):
    img = cv2.imread(img_directory+filename)
    cv2.imwrite(aug_directory+filename, img)
    post_crop_dim = img.shape[1] * 0.6
    cv2.imwrite(aug_directory+'1_'+filename, center_crop_vert(img,post_crop_dim))
    cv2.imwrite(aug_directory+'2_'+filename, center_crop_horz(img,post_crop_dim))

