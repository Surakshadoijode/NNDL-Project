import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import build_model
import matplotlib.pyplot as plt

# Load Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize & reshape
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Build model
model = build_model()

print("Training model...")
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_acc)

# Save model
model.save("fashion_mnist_cnn.keras")
print("Model saved as fashion_mnist_cnn.keras")

