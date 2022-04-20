import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from collections import Counter
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Object Detection
# Step 1
def return_boxed_cropped_img(filename):
    
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
    gray= Image.open('static/uploads/object_detection_cropped/'+cropped_img_path)
    gray = ImageOps.grayscale(gray)
    gray = np.array(gray)
    reconstructed_model = tf.keras.models.load_model("static/model_checkpoints/no_mask_model")
    gray=gray.reshape((64,64,1))
    gray=tf.expand_dims(gray, axis=0)
    pred=reconstructed_model.predict(gray)
    return pred, np.round(pred[0][0]), np.round(pred[0][1]), np.round(pred[0][2])

def return_top_half_image_classification(top_half_mask_image):
    gray= Image.open('static/uploads/top_half_masked/'+top_half_mask_image)
    gray = ImageOps.grayscale(gray)
    gray = np.array(gray)
    reconstructed_model = tf.keras.models.load_model("static/model_checkpoints/top_mask_model")
    gray=gray.reshape((32,64,1))
    gray=tf.expand_dims(gray, axis=0)
    pred=reconstructed_model.predict(gray)
    return pred, np.round(pred[0][0]), np.round(pred[0][1]), np.round(pred[0][2])

def return_bottom_half_image_classification(botton_half_mask_image):
    gray= Image.open('static/uploads/bottom_half_masked/'+botton_half_mask_image)
    gray = ImageOps.grayscale(gray)
    gray = np.array(gray)
    reconstructed_model = tf.keras.models.load_model("static/model_checkpoints/bottom_mask_model")
    gray=gray.reshape((32,64,1))
    gray=tf.expand_dims(gray, axis=0)
    pred=reconstructed_model.predict(gray)
    return pred, np.round(pred[0][0]), np.round(pred[0][1]), np.round(pred[0][2])

def return_three_fourths_image_classification(three_fourths_mask_image):
    gray= Image.open('static/uploads/three_fourths_masked/'+three_fourths_mask_image)
    gray = ImageOps.grayscale(gray)
    gray = np.array(gray)
    reconstructed_model = tf.keras.models.load_model("static/model_checkpoints/three_fourths_mask_model")
    gray=gray.reshape((48,64,1))
    gray=tf.expand_dims(gray, axis=0)
    pred=reconstructed_model.predict(gray)
    return pred, np.round(pred[0][0]), np.round(pred[0][1]), np.round(pred[0][2])


# Step 3 - Overall Emotion Classification.
def return_overall_classification(full_image_pred, top_half_mask_pred, bottom_half_mask_pred, three_fourths_mask_pred):
    labels={0:'Happy',1: 'Sad',2: 'Neutral'}
    all_classifications = []
    all_classifications.append(np.argmax(np.array(full_image_pred)))
    all_classifications.append(np.argmax(np.array(top_half_mask_pred)))
    all_classifications.append(np.argmax(np.array(bottom_half_mask_pred)))
    all_classifications.append(np.argmax(np.array(three_fourths_mask_pred)))

    # Best Model (Three-Fourths Model) Classification
    best_model_classification=labels[all_classifications[3]]

    # Majority Classification
    counts = Counter(all_classifications)
    print(counts)
    
    top_two = counts.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        majority_classification = labels[all_classifications[3]]
    majority_classification = labels[top_two[0][0]]

    return majority_classification, best_model_classification


if __name__ == "__main__":
    
    # Object Detection
    # Step 1
    return_filepath_boxed_img, return_filepath_cropped_img = return_boxed_cropped_img('sad_images_28.jpg')
    
    # Emotion Classification
    # Step 1. Front-End UI Display.
    return_filepath_top_half = return_top_half_mask_image(return_filepath_cropped_img)
    return_bottom_half_path = return_bottom_half_mask_image(return_filepath_cropped_img)
    return_three_fourths_path = return_three_fourths_mask_image(return_filepath_cropped_img)
    
    # Emotion Classification
    # Step 2. Pass back dictionary here containing percentage of Happy, Neutral, and Sad.
    full_image_pred, f_h, f_s, f_n = return_full_image_classification(return_filepath_cropped_img)
    top_half_mask_pred, t_h, t_s, t_n = return_top_half_image_classification(return_filepath_top_half)
    bottom_half_mask_pred, b_h, b_s, b_n = return_bottom_half_image_classification(return_bottom_half_path)
    three_fourths_mask_pred, tf_h, tf_s, tf_n = return_three_fourths_image_classification(return_three_fourths_path)
    
    # Step 3 - Overall Emotion Classification.
    majority_classification, best_model_classification = return_overall_classification(full_image_pred, 
                                                       top_half_mask_pred, 
                                                       bottom_half_mask_pred,
                                                       three_fourths_mask_pred)
    
    print('\n\n')
    print('Boxed Image Filepath: ', return_filepath_boxed_img)
    print('Cropped Image Filepath: ', return_filepath_cropped_img)
    print('Top Half Masked Image Filepath: ', return_filepath_top_half)
    print('Bottom Half Masked Image Filepath: ', return_bottom_half_path)
    print('Three Fourths Masked Image Filepath: ', return_three_fourths_path)
    
    print()
    print("Full Image - Happy:",f_h)
    print("Full Image - Sad:",f_s)
    print("Full Image - Neutral:",f_n)
    
    print()
    print("Top Half Mask - Happy:",t_h)
    print("Top Half Mask - Sad:",t_s)
    print("Top Half Mask - Neutral:",t_n)
    
    print()
    print("Bottom Half Mask - Happy:",b_h)
    print("Bottom Half Mask - Sad:",b_s)
    print("Bottom Half Mask - Neutral:",b_n)
    
    print()
    print("Three Fourths Mask - Happy:",tf_h)
    print("Three Fourths Mask - Sad:",tf_s)
    print("Three Fourths Mask - Neutral:",tf_n)
    
    print()
    print('The Majority Classification is: ', majority_classification)
    print('The Best Model (Three-Fourths) Classification is: ', best_model_classification)
    print('\n\n')