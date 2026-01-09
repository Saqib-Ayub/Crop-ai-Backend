from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import Dict
import uvicorn
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import traceback


app = FastAPI(title="Potato Disease Detection API")

# CORS middleware for React Native frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = "potato_leaf_efficientnetv2.tflite"  # Update with your model path
IMG_SIZE = 224  # Standard size, adjust based on your model
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Disease recommendations
RECOMMENDATIONS = {
    "Early Blight": {
        "description": "Early blight is caused by the fungus Alternaria solani. It appears as dark brown spots with concentric rings on older leaves.",
        "treatment": [
            "Remove and destroy infected plant debris",
            "Apply fungicides containing chlorothalonil or mancozeb",
            "Ensure proper plant spacing for air circulation",
            "Avoid overhead watering",
            "Rotate crops annually"
        ],
        "prevention": [
            "Use disease-resistant varieties",
            "Mulch around plants to prevent soil splash",
            "Water at the base of plants in the morning"
        ]
    },
    "Late Blight": {
        "description": "Late blight is caused by Phytophthora infestans. It causes water-soaked spots that turn brown and spreads rapidly in cool, wet conditions.",
        "treatment": [
            "Remove infected plants immediately",
            "Apply copper-based fungicides or systemic fungicides",
            "Destroy all infected tubers",
            "Improve drainage in the field",
            "Monitor weather conditions for disease-favorable periods"
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes",
            "Avoid planting near tomato crops",
            "Use resistant varieties when available",
            "Implement early warning systems"
        ]
    },
    "Healthy": {
        "description": "Your potato plant appears healthy! Continue good agricultural practices.",
        "treatment": [
            "No treatment needed"
        ],
        "prevention": [
            "Maintain regular monitoring",
            "Ensure balanced fertilization",
            "Practice crop rotation",
            "Keep the field free from weeds",
            "Maintain optimal soil moisture"
        ]
    }
}

# Global variable for model
interpreter = None

def load_model():
    global interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
    print("✅ TFLite model loaded")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img, dtype=np.float32)

        # EfficientNetV2 preprocessing (DO NOT divide by 255)
        img_array = preprocess_input(img_array)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # FP16 MODEL → convert input to float16
        img_array = img_array.astype(np.float16)

        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

def predict(image_array: np.ndarray) -> Dict:
    # Run Prediction on preprocessed image
    # Returns prediction label and confidence
    try:
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        if image_array.dtype != input_details[0]["dtype"]:

            image_array = image_array.astype(input_details[0]["dtype"])

        interpreter.set_tensor(input_details[0]['index'], image_array)

        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions = output[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                CLASS_NAMES[i]: float(predictions[i]) 
                for i in range(len(CLASS_NAMES))
            }
        }
    except Exception as e:
        print("❌ FULL ERROR TRACE:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Potato Disease Detection API",
        "status": "running",
        "model_loaded": interpreter is not None
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Main prediction endpoint
    Accepts image file and returns disease prediction with recommendations
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        processed_image = preprocess_image(image_bytes)
        
        # Predict
        prediction_result = predict(processed_image)
        
        # Get disease name and recommendations
        disease = prediction_result["predicted_class"]
        recommendation = RECOMMENDATIONS.get(disease, {})
        
        # Build response
        response = {
            "success": True,
            "prediction": {
                "disease": disease,
                "confidence": round(prediction_result["confidence"] * 100, 2),
                "all_predictions": {
                    k: round(v * 100, 2) 
                    for k, v in prediction_result["all_predictions"].items()
                }
            },
            "recommendation": recommendation
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/classes")
async def get_classes():
    """Return available disease classes"""
    return {
        "classes": CLASS_NAMES,
        "total": len(CLASS_NAMES)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": interpreter is not None,
        "model_path": MODEL_PATH,
        "image_size": IMG_SIZE,
        "classes": len(CLASS_NAMES)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)