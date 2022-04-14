import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import tensorflow as tf

# Object Detection
# Step 1
def return_boxed_cropped_img(filename, img=None):
    
    img= Image.open('static/uploads/originals/'+filename)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = ImageOps.grayscale(img)
    gray = np.array(gray)
    
    #Storing in the Dataframe
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        #startpoint, endpoint, color, thickness
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cropped_img= gray[y:y+h,x:x+w]
        cropped_image=cv2.resize(cropped_img, (64,64), interpolation = cv2.INTER_AREA)
    
    return_boxed_path = filename+"_boxed.png"
    return_cropped_path = filename+"_cropped.png"
    filepath_boxed_img = 'static/uploads/object_detection_boxed/'+return_boxed_path
    filepath_cropped_img = 'static/uploads/object_detection_cropped/'+return_cropped_path
    plt.imsave(filepath_boxed_img, gray)
    plt.imsave(filepath_cropped_img, cropped_image)
    
    return return_boxed_path, return_cropped_path

# Emotion Classification
# Step 1. Front-End UI Display.
def return_top_half_mask_image(cropped_img_path):
    gray= Image.open('static/uploads/object_detection_cropped/'+cropped_img_path)
    gray = np.array(gray)
    top_half = gray[32:,:]
    return_top_half_path = cropped_img_path+"_top_half.png"
    filepath_top_half = 'static/uploads/top_half_masked/'+return_top_half_path
    plt.imsave(filepath_top_half, top_half)
    return return_top_half_path
    
def return_bottom_half_mask_image(cropped_img_path):
    gray= Image.open('static/uploads/object_detection_cropped/'+cropped_img_path)
    gray = np.array(gray)
    bottom_half = gray[:32,:]
    return_bottom_half_path = cropped_img_path+"_bottom_half.png"
    filepath_bottom_half = 'static/uploads/bottom_half_masked/'+return_bottom_half_path
    plt.imsave(filepath_bottom_half, bottom_half)
    return return_bottom_half_path

def return_three_fourths_mask_image(cropped_img_path):
    gray= Image.open('static/uploads/object_detection_cropped/'+cropped_img_path)
    gray = np.array(gray)
    three_fourths = gray[16:,:]
    return_three_fourths_path = cropped_img_path+"_three_fourths.png"
    filepath_three_fourths = 'static/uploads/three_fourths_masked/'+return_three_fourths_path
    plt.imsave(filepath_three_fourths, three_fourths)
    return return_three_fourths_path


# Step 2. Pass back dictionary here containing percentage of Happy, Neutral, and Sad.
def return_full_image_classification(cropped_img_path):
    return -1

# def return_top_half_image_classification(top_half_mask_image):
#     return -1


# def return_bottom_half_image_classification(botton_half_mask_image):
#     return -1


# def return_three_fourths_image_classification(three_fourths_mask_image):
#     return -1



if __name__ == "__main__":
    
    # Object Detection
    # Step 1
    return_filepath_boxed_img, return_filepath_cropped_img = return_boxed_cropped_img('image.png')
    
    # Emotion Classification
    # Step 1. Front-End UI Display.
    return_filepath_top_half = return_top_half_mask_image(return_filepath_cropped_img)
    return_bottom_half_path = return_bottom_half_mask_image(return_filepath_cropped_img)
    return_three_fourths_path = return_three_fourths_mask_image(return_filepath_cropped_img)
    
    # Emotion Classification
    # Step 2. Pass back dictionary here containing percentage of Happy, Neutral, and Sad.
    return_full_image_classification(return_filepath_cropped_img)
    
    
    print(return_filepath_boxed_img)
    print(return_filepath_cropped_img)
    print(return_filepath_top_half)
    print(return_bottom_half_path)
    print(return_three_fourths_path)
        
        
# # Step 3 - Overall Emotion Classification.
# def return_overall_classification(full_classification, top_half_classification, bottom_half_classification,
#                                   three_fourths_half_classification):
#     return "happy"