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

# Class labels mapping
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
    # This block handles the POST request when a user uploads an image
    if request.method == 'POST':
        file = request.files.get('image') # Use .get() for safer access
        if file and file.filename != '':
            # Save the image with a unique name
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Preprocess the image
            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            # Predict
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            result = class_labels[predicted_class % 2]

            # --- KEY CHANGE ---
            # Instead of rendering, we redirect to a new 'result' page
            # We pass the prediction data as URL parameters
            return redirect(url_for('result', prediction_result=result, image_filename=filename))
            
    # If it's a GET request, just show the upload page
    return render_template('predict.html')

# --- NEW ROUTE ---
# This route is only for displaying the result page.
@app.route('/result')
def result():
    # Get the data passed from the redirect
    prediction = request.args.get('prediction_result')
    filename = request.args.get('image_filename')
    
    # Create the full URL for the image to be used in the template
    image_url = url_for('static', filename='uploads/' + filename)

    # Render the result template with the data
    return render_template('portfolio-details.html', result=prediction, image_url=image_url)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Contact form submitted by {name} ({email}): {message}")
        return redirect(url_for('contact'))
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
