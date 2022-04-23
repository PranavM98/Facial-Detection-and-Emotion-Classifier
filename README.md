# Physical Layer Data Augmentation Techniques for Face Recognition and Face Emotion Classification Data

## Abstract
In this project, we explore various physical layer augmentation techniques such as top-half mask, bottom-half mask, one-fourth mask, along with an image data augmentation technique, sharpness kernel, on the Face Recognition and Emotion Classification dataset using a VGG-16 model architecture. We learn that physical layer augmentation and data augmentation, in general, boosts model performance and reduces overfitting compared to the baseline. Specifically, masking one-fourth of the pixels in the physical layer, results in an accuracy of 69.76% similar to that of the baseline (no-mask) model of 72.31%. This shows us that 1/4th of the pixels can be discarded in a Face Emotion Classification experiment while still maintaining near baseline accuracy.  

## Data Process
<img width="708" alt="Process" src="https://user-images.githubusercontent.com/26104722/164836570-c45a12ed-8eea-4ae6-83f7-efd0ead6a3f7.png">

## Steps to Run Flask Web Application

1. Git Clone the repository onto your local.
2. Download the trained model checkpoints from this google drive location: https://drive.google.com/drive/folders/1fDAPtVYReKV1K1w-d7iTSguNSeE5ZOkH?usp=sharing
3. Specifically, download the "bottom_mask_model", "top_mask_model", "no_mask_model", and "three_fourths_mask_model" folders from this google drive location.
4. Once downloaded, move all of these folders under a new folder called "model_checkpoints".
5. Move this new "model_checkpoints" folder under "Facial-Detection-and-Emotion-Classifier/10_Code/03_Flask_App/static/"
6. Create a new Anaconda (conda) environment and install packages listed in: https://github.com/PranavM98/Facial-Detection-and-Emotion-Classifier/blob/main/10_Code/03_Flask_App/requirements.txt
7. Navigate to "Facial-Detection-and-Emotion-Classifier/10_Code/03_Flask_App/" on your local, and run "python3 app.py" to run the Flask App.
