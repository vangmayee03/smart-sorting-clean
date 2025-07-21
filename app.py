
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid

app = Flask(__name__)

# --- Load Your Model ---
# Make sure your model file is in the same directory as this script
model = load_model('healthy_vs_rotten.h5')

# --- Configuration for File Uploads ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create the folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the class labels your model will predict
class_labels = {
    0: 'HEALTHY',
    1: 'ROTTEN'
}

# --- Page Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# This route handles both showing the upload form and processing the uploaded image
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # This block runs when the user clicks the "Predict" button
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        if file:
            # 1. Save the uploaded file
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 2. Preprocess the image for your model
            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            # 3. Get a prediction from the model
            prediction = model.predict(image)
            predicted_class_index = np.argmax(prediction)
            result_label = class_labels.get(predicted_class_index, "UNKNOWN")

            # 4. Redirect to the result page with the prediction data
            return redirect(url_for('result', prediction_result=result_label, image_filename=filename))

    # If it's a GET request, just show the upload page
    return render_template('predict.html')


# This new route is dedicated to showing the result page
@app.route('/result')
def result():
    # Get the prediction data sent from the predict() function
    prediction = request.args.get('prediction_result')
    filename = request.args.get('image_filename')
    
    # Create the URL for the uploaded image so the HTML can display it
    image_url = url_for('static', filename='uploads/' + filename)

    # Render your 'portfolio-details.html' page with the data
    return render_template('portfolio-details.html', result=prediction, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
