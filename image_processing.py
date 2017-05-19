import cv2
import numpy as np

def contrast_image(bgr_image):
    #-----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #cv2.imshow('final', final)
    return final

def random_shadow(image):
    top_y = image.shape[1] * np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1] * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    result = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return result

def show_image_comparison(image1, image2):

    cv2.imshow("Image1",image1)
    cv2.imshow("Image2",image2)

#image = cv2.imread('image/shadow_trees.jpg')
#show_image_comparison(image, contrast_image(image))

#image = cv2.imread('image/clear_road.jpg')
#show_image_comparison(image, random_shadow(image))

#cv2.waitKey(0)