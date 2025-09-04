import tensorflow as tf

# Load your trained model (e.g. .h5 or SavedModel folder)
model = tf.keras.models.load_model("tomato_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save it
with open("tomato_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as tomato_model.tflite")
