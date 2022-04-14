from flask import Flask, flash, request, render_template, redirect, url_for
import urllib.request
import os
from werkzeug.utils import secure_filename
from model.custom_functions import *

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
        boxed_img_path, cropped_img_path = return_boxed_cropped_img(filename)
        
        # Emotion Classification
        # Step 1. Front-End UI Display.
        top_half_mask_image = return_top_half_mask_image(cropped_img_path)
        botton_half_mask_image = return_bottom_half_mask_image(cropped_img_path)
        three_fourths_mask_image = return_three_fourths_mask_image(cropped_img_path)
        
        # Step 2. Pass back dictionary here containing percentage of Happy, Neutral, and Sad.
        full_classification = return_full_image_classification(cropped_img_path)
        top_half_classification = return_top_half_image_classification(top_half_mask_image)
        bottom_half_classification = return_bottom_half_image_classification(botton_half_mask_image)
        three_fourths_half_classification = return_three_fourths_image_classification(three_fourths_mask_image)
        
        # Step 3 - Overall Emotion Classification.
        final_classification = return_overall_classification(full_classification, 
                                                             top_half_classification, 
                                                             bottom_half_classification,
                                                            three_fourths_half_classification)
        
        return render_template('home-post.html', 
                              # Object Detection - Step 1
                              boxed_img_path=boxed_img_path,
                              cropped_img_path=cropped_img_path,
                              
                              # Emotion Classification - Step 1
                              top_half_mask_image=top_half_mask_image,
                              botton_half_mask_image=botton_half_mask_image,
                              three_fourths_mask_image=three_fourths_mask_image,
                              
                              # Emotion Classification - Step 2
                              full_classification=full_classification,
                              top_half_classification=top_half_classification,
                              bottom_half_classification=bottom_half_classification,
                              three_fourths_half_classification=three_fourths_half_classification,
                              
                              # Emotion Classification - Step 3
                              final_classification=final_classification, 
                              filename=filename)
    
    # Error Handling. Only allowed file extensions.
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)