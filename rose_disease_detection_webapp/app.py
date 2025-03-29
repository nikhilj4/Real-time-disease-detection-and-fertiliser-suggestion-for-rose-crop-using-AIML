from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')
class_labels = ['Healthy', 'Diseased']
IMAGE_SIZE = (224, 224)

# Function to classify disease stage
def classify_disease_stage(prediction):
    healthy_percent = prediction[0][0] * 100
    diseased_percent = prediction[0][1] * 100
    
    if healthy_percent >= 90:
        stage = "Healthy"
        suggestion = "Keep monitoring the plant regularly and ensure proper watering and sunlight."
    elif diseased_percent <= 20:
        stage = "Beginning Stage"
        suggestion = "Remove affected leaves and apply a mild fungicide. Ensure the plant has proper ventilation."
    elif 20 < diseased_percent <= 60:
        stage = "Intermediate Stage"
        suggestion = "Prune infected areas and apply a recommended fungicide like Copper-based sprays or Neem oil. Avoid overwatering."
    else:
        stage = "Severe Stage"
        suggestion = "Remove heavily infected parts or, if necessary, the entire plant to prevent spread. Use a strong fungicide and disinfect surrounding soil."
    
    return healthy_percent, diseased_percent, stage, suggestion

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        if file:
            img = Image.open(file.stream).convert('RGB')
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            healthy_percent, diseased_percent, stage, suggestion = classify_disease_stage(prediction)

            return render_template('index.html',
                                   healthy=f"{healthy_percent:.2f}%",
                                   diseased=f"{diseased_percent:.2f}%",
                                   stage=stage,
                                   suggestion=suggestion)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
