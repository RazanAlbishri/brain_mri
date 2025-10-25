import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image

app = FastAPI(title="MRI Brain Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "MRI_brain_Model_FIXED.h5"

def load_model_with_fix(model_path):
    try:
        print("Attempting to load model normally...")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully with normal method")
        return model
    except Exception as e:
        print(f"Model load error: {e}")
        raise e

try:
    model = load_model_with_fix(MODEL_PATH)
    print("✅ Model successfully loaded and ready for predictions!")

    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    print(f" Model test prediction shape: {prediction.shape}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise e

# ✅ Real medical brain tumor classes
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
print(f"Class labels: {class_labels}")

@app.get("/")
def home():
    return {
        "message": "MRI Brain Classifier API is running ✅",
        "status": "healthy",
        "model_loaded": True,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "class_labels": class_labels,
        "input_shape": model.input_shape
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))

        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        predicted_label = class_labels[predicted_class_idx]

        all_confidences = {
            class_labels[i]: f"{float(predictions[0][i] * 100):.2f}%"
            for i in range(len(class_labels))
        }

        return JSONResponse({
            "prediction": predicted_label,
            "confidence": f"{confidence:.2f}%",
            "all_predictions": all_confidences,
            "status": "success"
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Proper entry point for deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
