from flask import Flask, flash, request, render_template, redirect, url_for
import urllib.request
import os
from werkzeug.utils import secure_filename
from model.custom_functions import *

app = Flask(__name__, static_url_path='/static')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'static/uploads/'
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
        classification_result = return_image_classification(filename)
        return render_template('home-post.html', classification_result=classification_result, filename=filename)
    
    # Error Handling. Only allowed file extensions.
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)