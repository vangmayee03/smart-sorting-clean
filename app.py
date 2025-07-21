from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)
model = load_model('healthy_vs_rotten.h5')

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Class labels mapping (simplified for this example)
class_labels = {
    0: 'HEALTHY',
    1: 'ROTTEN'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save the image
            filename = str(uuid.uuid4()) + '.jpg'
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess the image
            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            # Predict
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            result = class_labels[predicted_class % 2]  # 0: HEALTHY, 1: ROTTEN

            return render_template('portfolio-details.html', result=result, image_url=url_for('static', filename='uploads/' + filename))
    return render_template('predict.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # You can log the message or email it
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Contact form submitted by {name} ({email}): {message}")
        return redirect(url_for('contact'))
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
