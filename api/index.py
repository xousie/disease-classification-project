from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = FastAPI(title="Skin Disease Prediction API")

# Load the Model (Do this ONCE at startup)
try:
    densenet_model = load_model('./models/densenet.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    densenet_model = None  # Important: Handle model loading failure

CLASS_NAMES = ['Karsinoma', 'Kulit Sehat', 'Melanoma']


def preprocess_image(image):
    """
    Preprocesses the image to be suitable for the DenseNet model.

    Args:
        image (PIL.Image.Image): The image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image as a numpy array.
    """
    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
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


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile):
    """
    Endpoint for predicting the class of an uploaded image.
    """

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        preprocessed_image = preprocess_image(image)

        if densenet_model is not None:
            predictions, predicted_class = predict(densenet_model, preprocessed_image)

            return JSONResponse({
                "filename": file.filename,
                "content_type": file.content_type,
                "predicted_class": predicted_class,
                "probabilities": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
            })
        else:
            raise HTTPException(status_code=500, detail="Model loading failed.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
