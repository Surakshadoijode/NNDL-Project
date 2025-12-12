# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # Load model
# model = keras.models.load_model("fashion_mnist_cnn.keras")
# print("Model loaded.")

# # Load test data
# fashion_mnist = keras.datasets.fashion_mnist
# (_, _), (test_images, test_labels) = fashion_mnist.load_data()

# # Prepare image
# test_images = test_images / 255.0
# img = test_images[0].reshape(1, 28, 28, 1)

# prediction = model.predict(img)
# pred_class = np.argmax(prediction)

# # Show result
# plt.imshow(test_images[0], cmap='gray')
# plt.title(f"Prediction: {class_names[pred_class]}")
# plt.show()

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the saved model
# model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

# # Load data again (test images only)
# mnist = tf.keras.datasets.fashion_mnist
# (_, _), (test_images, test_labels) = mnist.load_data()

# test_images = test_images / 255.0

# # Class names for Fashion MNIST
# class_names = [
#     "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
#     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
# ]

# # Show and predict 5 images
# for i in range(5):
#     img = test_images[i].reshape(1, 28, 28, 1)
#     prediction = model.predict(img)
#     pred_class = np.argmax(prediction)

#     plt.imshow(test_images[i], cmap='gray')
#     plt.title(f"Predicted: {class_names[pred_class]}")
#     plt.axis('off')
#     plt.show()

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # Class names
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # Load model
# model = tf.keras.models.load_model("fashion_mnist_cnn.keras")

# # ----------- GIVE YOUR IMAGE HERE ------------
# image_path = "images/ai-generated-8557635_1280.jpg"   # <-- Change this to your image file
# # ---------------------------------------------

# # Load and preprocess image
# img = Image.open(image_path).convert("L")  # convert to grayscale
# img = img.resize((28, 28))                # resize to 28x28
# img_array = np.array(img) / 255.0
# img_array = img_array.reshape(1, 28, 28, 1)

# # Predict
# prediction = model.predict(img_array)
# class_id = np.argmax(prediction)

# # Show output
# plt.imshow(img, cmap="gray")
# plt.title(f"Predicted: {class_names[class_id]}")
# plt.show()

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import sys

# Class names of Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load model
model = tf.keras.models.load_model("fashion_mnist_cnn.keras")
print("Model loaded.")

# If user gives an image: python predict.py img.jpg
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    print("Using your image:", image_path)

    # --- PREPROCESS REAL IMAGE ---
    img = Image.open(image_path).convert("L")   # convert to grayscale
    img = ImageOps.invert(img)                 # invert colors (like Fashion-MNIST)
    img = img.resize((28, 28))                 # resize to 28x28

    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    prediction = model.predict(img_arr)
    pred_class = np.argmax(prediction)

    # Show result
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {class_names[pred_class]}")
    plt.show()

else:
    print("No image given. Predicting from test dataset...")

    # Load Fashion MNIST test set
    mnist = tf.keras.datasets.fashion_mnist
    (_, _), (test_images, test_labels) = mnist.load_data()

    test_images = test_images / 255.0

    # Predict 5 sample test images
    for i in range(5):
        img = test_images[i].reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)

        plt.imshow(test_images[i], cmap='gray')
        plt.title(f"Predicted: {class_names[pred_class]}")
        plt.show()
