import os
import cv2
from tensorflow.keras.datasets import fashion_mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Create folders
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/test", exist_ok=True)

# Save training images
for i in range(len(X_train)):
    img = X_train[i]
    cv2.imwrite(f"dataset/train/image_{i}_label_{y_train[i]}.png", img)

# Save test images
for i in range(len(X_test)):
    img = X_test[i]
    cv2.imwrite(f"dataset/test/image_{i}_label_{y_test[i]}.png", img)

print("Images saved successfully!")
