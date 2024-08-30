import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('model.h5')

# Load and preprocess the input image
img_path = '/home/muhd/Desktop/python-proj/Satelite-deeplearning/artifacts/data_ingestion/data/water/SeaLake_1.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size according to your model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image if your model was trained with normalized images

# Make predictions
predictions = model.predict(img_array)

# Print the predicted class
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class}")

# If you have a class labels mapping
class_labels = ['cloudy', 'desert', 'water','green_area']  # Replace with your actual class labels
print(f"Predicted label: {class_labels[predicted_class[0]]}")
