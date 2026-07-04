
from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import sqlite3
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

# ====================================
# FLASK APP
# ====================================

app = Flask(__name__)

app.secret_key = 'plantdiseasekey'

# ====================================
# FOLDERS
# ====================================

UPLOAD_FOLDER = 'static/uploads'

GRAY_FOLDER = 'static/processed/gray'
EDGE_FOLDER = 'static/processed/edge'
THRESHOLD_FOLDER = 'static/processed/threshold'
SHARPEN_FOLDER = 'static/processed/sharpen'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====================================
# LOAD MODEL
# ====================================

model = load_model('model/plant_disease_model.h5')

# ====================================
# CLASS LABELS
# ====================================

classes = {
    0: 'Healthy',
    1: 'Miner',
    2: 'Phoma',
    3: 'Rust'
}

# ====================================
# REMEDIES
# ====================================

remedies = {

    'Healthy': [
        'Plant leaf is healthy',
        'No disease detected',
        'Maintain regular care'
    ],

    'Miner': [
        'Remove infected leaves',
        'Use neem oil spray',
        'Apply suitable insecticide'
    ],

    'Phoma': [
        'Use fungicide spray',
        'Avoid excess watering',
        'Remove infected areas'
    ],

    'Rust': [
        'Apply copper fungicide',
        'Improve air circulation',
        'Avoid leaf moisture'
    ]
}

# ====================================
# DATABASE
# ====================================

conn = sqlite3.connect('users.db')

conn.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
)
''')

conn.close()

# ====================================
# HOME PAGE
# ====================================

@app.route('/')
def home():
    return render_template('index.html')

# ====================================
# REGISTER
# ====================================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')

        cursor = conn.cursor()

        cursor.execute(
            'INSERT INTO users(username, password) VALUES(?, ?)',
            (username, password)
        )

        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    return render_template('register.html')

# ====================================
# LOGIN
# ====================================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')

        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM users WHERE username=? AND password=?',
            (username, password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:

            session['username'] = username

            return redirect('/')

        else:
            return 'Invalid Username or Password'

    return render_template('login.html')

# ====================================
# IMAGE UPLOAD PAGE
# ====================================

@app.route('/image')
def image_upload():
    return render_template('index.html')

# ====================================
# IMAGE PREDICTION
# ====================================

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('filename')

    if file is None:
        return "No file uploaded"

    # Save uploaded image
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # ====================================
    # IMAGE PROCESSING
    # ====================================

    img = cv2.imread(filepath)

    if img is None:
        return "Image not loaded properly"
    # Gray Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_path = os.path.join(
        GRAY_FOLDER,
        'gray_' + file.filename
    )

    cv2.imwrite(gray_path, gray)

    # Edge Detection
    edge = cv2.Canny(gray, 100, 200)

    edge_path = os.path.join(
        EDGE_FOLDER,
        'edge_' + file.filename
    )

    cv2.imwrite(edge_path, edge)

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY
    )

    threshold_path = os.path.join(
        THRESHOLD_FOLDER,
        'threshold_' + file.filename
    )

    cv2.imwrite(threshold_path, thresh)

    # Sharpen
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    sharpen = cv2.filter2D(img, -1, kernel)

    sharpen_path = os.path.join(
        SHARPEN_FOLDER,
        'sharpen_' + file.filename
    )

    cv2.imwrite(sharpen_path, sharpen)

    # ====================================
    # CNN PREDICTION
    # ====================================

    test_image = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    test_image = image.img_to_array(test_image)

    test_image = test_image / 255.0

    test_image = np.expand_dims(test_image, axis=0)

    prediction = model.predict(test_image)

    print(prediction)

    pred = prediction[0]

    healthy = pred[0]
    miner = pred[1]
    phoma = pred[2]
    rust = pred[3]

    max_index = np.argmax(pred)

    confidence = pred[max_index] * 100

    # ====================================
    # SMART HEALTHY DETECTION
    # ====================================

    # ====================================
    # IMPROVED SMART PREDICTION
    # ====================================

    if healthy >= 0.25:

        status = 'Healthy'

    elif rust >= 0.70:

        status = 'Rust'

    elif phoma >= 0.45:

        status = 'Phoma'

    elif miner >= 0.45:

        status = 'Miner'

    else:

        status = classes[max_index]


    accuracy = round(confidence, 2)

    print(status)

    Remedies1 = remedies[status]

    # ====================================
    # SHOW RESULT
    # ====================================

    return render_template(

        'result.html',

        status=status,

        accuracy=accuracy,

        Remedies1=Remedies1,

        ImageDisplay=filepath,

        ImageDisplay1=gray_path,

        ImageDisplay2=edge_path,

        ImageDisplay3=threshold_path,

        ImageDisplay4=sharpen_path

    )

@app.route('/graph')
def graph():
    return render_template('graph.html')

# ====================================
# LOGOUT
# ====================================

@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for('home'))

# ====================================
# MAIN
# ====================================

if __name__ == '__main__':
    app.run(debug=True)