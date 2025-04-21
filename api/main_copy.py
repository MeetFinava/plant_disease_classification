from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json


app= FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



MODEL=tf.keras.models.load_model("../saved_models/finalmodel.h5",compile=True,custom_objects=None)

CLASS_NAMES=[
'Apple___brown_spot',
 'Apple___healthy',
 'Apple___scab',
 'Corn___common_rust',
 'Corn___gray_leaf_spot',
 'Corn___healthy',
 'Corn___northern_leaf_blight',
 'Grape___Leaf_blight',
 'Grape___black_measles',
 'Grape___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Strawberry___healthy',
 'Strawberry___leaf_scorch',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_healthy',
 'Watermelon___downy_mildew',
 'Watermelon___healthy',
 'Watermelon___mosaic_virus']


def read_file_as_image(data) -> np.ndarray:
        # Ensure that the file is a valid image
        image = Image.open(BytesIO(data))
        # # image = image.convert("RGB")  # Convert to RGB (if not already)
        # image = image.resize((128, 128))  # Resize to match the model's input
        # image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        return image
  

    
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   
        image = await file.read()
        img_array = read_file_as_image(image)
        img_batch = np.expand_dims(img_array, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)