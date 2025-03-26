import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('grass_classifier_model.keras')

# Load and preprocess the image
img = load_img('grass4.jpeg', target_size=(224, 224))  # Load the image and resize to 224x224
img_array = img_to_array(img)  # Convert image to a NumPy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image just like the training data

# Make a prediction
prediction = model.predict(img_array)

# Display the prediction result
print(f"Prediction (0: not grass, 1: grass): {prediction[0][0]}")

if prediction[0][0] > 0.5:
    print("Prediction: Grass")
else:
    print("Prediction: Not Grass")
