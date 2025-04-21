from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:19006"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/finalmodel.h5", compile=True, custom_objects=None)

CLASS_NAMES = [
    'Apple_brown_spot', 'Apple_healthy', 'Apple_scab',
    'Corn_common_rust', 'Corn_gray_leaf_spot', 'Corn_healthy', 'Corn_northern_leaf_blight',
    'Grape_Leaf_blight', 'Grape_black_measles', 'Grape_healthy',
    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
    'Strawberry_healthy', 'Strawberry_leaf_scorch',
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy',
    'Watermelon_downy_mildew', 'Watermelon_healthy', 'Watermelon_mosaic_virus'
]

# Sample medicine database (you can expand this or load from JSON file)
DISEASE_MEDICINE_MAP = {
    "Apple_brown_spot": {
        "name": "Captan",
        "description": "Controls brown spot in apples effectively.",
        "brand": "FungoStop"
    },
    "Apple_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Apple_scab": {
        "name": "Myclobutanil",
        "description": "Systemic fungicide effective for scab prevention.",
        "brand": "Rally"
    },
    "Corn_common_rust": {
        "name": "Propiconazole",
        "description": "Prevents and treats rust in corn.",
        "brand": "Tilt"
    },
    "Corn_gray_leaf_spot": {
        "name": "Azoxystrobin",
        "description": "Used for controlling gray leaf spot in corn.",
        "brand": "Quadris"
    },
    "Corn_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Corn_northern_leaf_blight": {
        "name": "Pyraclostrobin",
        "description": "Controls northern leaf blight in corn.",
        "brand": "Headline"
    },
    "Grape_Leaf_blight": {
        "name": "Zineb",
        "description": "Protects grapes from leaf blight.",
        "brand": "Dithane Z-78"
    },
    "Grape_black_measles": {
        "name": "Trifloxystrobin",
        "description": "Effective against black measles in grapes.",
        "brand": "Flint"
    },
    "Grape_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Potato_Early_blight": {
        "name": "Mancozeb",
        "description": "Controls early blight effectively.",
        "brand": "BlightGuard"
    },
    "Potato_Late_blight": {
        "name": "Chlorothalonil",
        "description": "Prevents and treats late blight in potatoes.",
        "brand": "LateXPro"
    },
    "Potato_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Strawberry_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Strawberry_leaf_scorch": {
        "name": "Copper hydroxide",
        "description": "Used to treat bacterial leaf scorch in strawberries.",
        "brand": "Kocide"
    },
    "Tomato_Early_blight": {
        "name": "Dithane M-45",
        "description": "Controls early blight in tomato leaves.",
        "brand": "TomSafe"
    },
    "Tomato_Late_blight": {
        "name": "Ridomil Gold",
        "description": "Strong protection from late blight.",
        "brand": "BlightShield"
    },
    "Tomato_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Watermelon_downy_mildew": {
        "name": "Metalaxyl-M",
        "description": "Effective against downy mildew in watermelon.",
        "brand": "Revus"
    },
    "Watermelon_healthy": {
        "name": "No treatment needed",
        "description": "Your plant is healthy.",
        "brand": "-"
    },
    "Watermelon_mosaic_virus": {
        "name": "Imidacloprid",
        "description": "Prevents viral spread by targeting aphid vectors.",
        "brand": "Confidor"
    }
}


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
 
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Query("en")):
    image = await file.read()
    img_array = read_file_as_image(image)
    img_batch = np.expand_dims(img_array, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    medicine = DISEASE_MEDICINE_MAP.get(predicted_class, {
        "name": "No specific medicine found",
        "description": "Please consult an expert for advice.",
        "brand": "N/A"
    })

    return {
        'disease': predicted_class,
        'confidence': float(confidence),
        'medicine': medicine
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
