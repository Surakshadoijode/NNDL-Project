import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

model = tf.keras.models.load_model("fashion_mnist_cnn.keras")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion MNIST Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")   # grayscale
    img = ImageOps.invert(img)                     # invert important!
    img = img.resize((28, 28))                     # resize

    img_arr = np.array(img) / 255.0                # normalize
    img_arr = img_arr.reshape(1, 28, 28, 1)        # reshape

    prediction = model.predict(img_arr)
    pred_class = np.argmax(prediction)

    st.image(img, caption=f"Predicted: {class_names[pred_class]}")

