import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2  #opencv

# Load the Model
@st.cache_resource
def load_densenet_model():
    try:
        model = load_model('./models/densenet.keras')  # Path might need adjustment
        print("Model loaded successfully!") #debug
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None  # Handle the error gracefully.

densenet_model = load_densenet_model()

CLASS_NAMES = ['Karsinoma', 'Kulit Sehat', 'Melanoma']


def preprocess_image(image):
    """
    Preprocesses the image to be suitable for the DenseNet model.

    Args:
        image (PIL.Image.Image): The image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image as a numpy array.
    """
    img = image.resize((256, 256))  # Resize
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(model, preprocessed_image):
    """
    Predicts the class of the image using the provided model.

    Args:
        model (tensorflow.keras.models.Model): The trained model to use for prediction.
        preprocessed_image (numpy.ndarray): The preprocessed image as a numpy array.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The raw prediction probabilities from the model.
            - str: The predicted class name (e.g., 'healthy', 'disease1', 'disease2').
    """

    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    return predictions, predicted_class


# Streamlit App
st.title("Skin Disease Prediction App")
st.write("Upload an image of a skin lesion for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
         # Read the image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


        preprocessed_image = preprocess_image(image)


        if densenet_model is not None:
            predictions, predicted_class = predict(densenet_model, preprocessed_image)

            st.subheader("Prediction Results:")
            st.write(f"Predicted Class: **{predicted_class}**")

            # Display probabilities (optional)
            st.write("Probabilities:")
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"- {class_name}: {predictions[0][i]:.4f}")
        else: # if model did not load
            st.error("Model loading failed. Check the console for errors")



    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Check the image format.  Ensure the model is loaded correctly.")


