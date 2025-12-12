import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")
print("Model loaded successfully!")

# Load Fashion MNIST test dataset
mnist = tf.keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = mnist.load_data()

# Normalize & reshape (same as in training)
test_images = test_images / 255.0
test_images = test_images.reshape(-1, 28, 28, 1)

# Evaluate accuracy
loss, acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", acc)
print("Accuracy (%):", acc * 100)
