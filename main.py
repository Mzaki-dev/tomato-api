from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = FastAPI()

# Allow your React Native app to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later to your app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = keras.models.load_model("model_epoch_60.keras", compile=False)

# Class names
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Friendly names
friendly_names = {
    'Tomato___Bacterial_spot': 'Bacterial Spot',
    'Tomato___Early_blight': 'Early Blight',
    'Tomato___Late_blight': 'Late Blight',
    'Tomato___Leaf_Mold': 'Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider Mites',
    'Tomato___Target_Spot': 'Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Mosaic Virus',
    'Tomato___healthy': 'Healthy'
}

# Directory for temporary uploads
PROCESSED_DIR = "processed_images"
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Save uploaded image
        filename = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join(PROCESSED_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        # Preprocess
        img = image.load_img(save_path, target_size=(300, 300), color_mode="rgb")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(predictions))

        return {
            "prediction": friendly_names.get(predicted_class, predicted_class),
            "confidence": round(confidence * 100, 2),
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
