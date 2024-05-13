import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def train_model(x_train, y_train, epochs=10, save_path='handwritten.keras'):
    """
    Train a model using the given training data.

    Parameters:
        x_train (numpy.ndarray): Input training data.
        y_train (numpy.ndarray): Target training data.
        epochs (int): Number of epochs for training. Default is 10.
        save_path (str): File path to save the trained model. Default is 'handwritten.keras'.

    Returns:
        tf.keras.models.Model: Trained model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=epochs)

    print(model.summary())
    model.save(save_path)

    return model

def check_digits(model, folder_path='digits'):
    """
    Check digits in images using the trained model.

    Parameters:
        model (tf.keras.models.Model): Trained model.
        folder_path (str): Path to the folder containing digit images. Default is 'digits'.
    """
    image_number = 0
    while os.path.isfile(f"{folder_path}/digit{image_number}.png"):
        try:
            img = cv2.imread(f"{folder_path}/digit{image_number}.png")[:, :, 0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Resolution is not correct")
        finally:
            image_number += 1

def create_model_and_check_digits():
    """
    Create and train a model, then check digits in images.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (_, _) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)

    model = train_model(x_train, y_train)
    check_digits(model)

if __name__ == "__main__":
    create_model_and_check_digits()