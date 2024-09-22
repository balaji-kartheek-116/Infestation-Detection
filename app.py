import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels only once
@st.cache_resource
def load_model_and_labels(model_path, labels_path):
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()
    return model, class_names

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    # Resize the image and crop from the center
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    # Convert the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image (to match the model's expected input)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Create a batch of one image
    data = np.ndarray(shape=(1, *target_size, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# Main Streamlit app
st.title("Infestation Detection System")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model and labels
    model, class_names = load_model_and_labels("keras_model.h5", "labels.txt")

    # Preprocess the image
    image_data = preprocess_image(image)

    # Predict
    with st.spinner("Analyzing the image..."):
        prediction = model.predict(image_data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

    # Display the prediction and confidence score
    st.write(f"Prediction: **{class_name}**")
    st.write(f"Confidence Score: **{confidence_score:.2f}**")

    if class_name.lower() == "mouse":
        st.success("Mouse Detected!")
    elif class_name.lower() == "earpods":
        st.success("EarPods Detected!")
    elif class_name.lower() == "mobile":
        st.success("Mobile Detected!")
    else:
        st.error("No Device Detected!")
