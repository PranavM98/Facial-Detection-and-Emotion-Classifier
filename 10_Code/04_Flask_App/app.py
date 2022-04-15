from flask import Flask, flash, request, render_template, redirect, url_for, session
import urllib.request
import os
from werkzeug.utils import secure_filename
from custom_functions import *

app = Flask(__name__, static_url_path='/static')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'static/uploads/originals/'
app.secret_key = "face-emotion-detection-secret-key-pranav-ashish-duke-mids"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
    
@app.route('/')
def my_form():
    return render_template('home.html')


@app.route('/', methods=['POST'])
def home_post():
    
    # Error handling. No file attached.
    if 'file' not in request.files:
        flash('No file part')
        return render_template('home.html')
    
    # Error handling. Incomplete Path
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return render_template('home.html')
    
    # Good Scenario.
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Object Detection
        # Step 1
        boxed_img_path, return_filepath_cropped_img = return_boxed_cropped_img(filename)
        session['return_filepath_cropped_img'] = return_filepath_cropped_img
        session['filename'] = filename
        
        return render_template('home-post.html', 
                              # Object Detection - Step 1
                              filename=filename,
                              boxed_img_path=boxed_img_path,
                              cropped_img_path=return_filepath_cropped_img)
    
    # Error Handling. Only allowed file extensions.
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return render_template('home.html')
        
@app.route('/emotion_classification')
def emotion_classification():
    
    # Pull Session Variable
    return_filepath_cropped_img = session['return_filepath_cropped_img']
    filename = session['filename']
    
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
        
    return render_template('emotion-classification.html', 
                          # Emotion Classification - Step 2
                          filename = filename,
                          return_filepath_cropped_img=return_filepath_cropped_img,
                          return_filepath_top_half=return_filepath_top_half,
                          return_bottom_half_path=return_bottom_half_path,
                          return_three_fourths_path=return_three_fourths_path,

                          f_h=f_h,
                          f_s=f_s,
                          f_n=f_n,
                           
                          t_h=t_h,
                          t_s=t_s,
                          t_n=t_n,
                           
                          b_h=b_h,
                          b_s=b_s,
                          b_n=b_n,
                           
                          tf_h=tf_h,
                          tf_s=tf_s,
                          tf_n=tf_n,
                           
                          majority_classification=majority_classification,
                          best_model_classification=best_model_classification 
                          )
        

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)