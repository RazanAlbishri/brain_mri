import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
from tensorflow.keras.applications.densenet import preprocess_input

# FastAPI app initialization
app = FastAPI(
    title="AI Medical Assistant - Brain MRI Classifier",
    description="Upload a brain MRI image and get tumor classification results",
    version="1.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Path
local_model_path = "brain_model.keras"

if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"⚠ Model file not found at: {local_model_path}")

# Load Model
model = tf.keras.models.load_model(local_model_path, compile=False)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
display_names = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

# Preprocess incoming images
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Prediction Endpoint
@app.post("/predict")
async def predict_brain(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Only JPG/PNG allowed.")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array)
    pred_idx = int(np.argmax(predictions))

    predicted_raw = class_names[pred_idx]
    predicted_class = display_names[predicted_raw]
    confidence = float(np.max(predictions)) * 100

    message = (
        "No tumor detected ✅"
        if predicted_raw == "notumor"
        else f"Possible {predicted_class} tumor detected — please consult a specialist 🔍"
    )

    return JSONResponse(content={
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "message": message
    })

# Root Endpoint
@app.get("/")
def root():
    return {"message": "✅ API is running successfully — Brain MRI Classifier"}
