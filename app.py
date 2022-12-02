import flask as fk
import tensorflow as tf
from tensorflow import keras
from keras import layers
# import base64
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import pickle
# Use pickle to load in the pre-trained model.
# with open(f'model/trained_model.h5', 'rb') as f:
#     model = pickle.load(f)
# app = flask.Flask(__name__, template_folder='templates')
# @app.route('/')
# def main():
#     return(flask.render_template('main.html'))
# if __name__ == '__main__':
#     app.run()


app = fk.Flask(__name__, template_folder='templates')
model = tf.keras.models.load_model("model/trained_model.h5")
target_img = os.path.join(os.getcwd(), 'images')


@app.route('/')
def main():
    return fk.render_template('main.html')


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'JPG'])

class_names = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple__Cedar_apple_rust',
               'Apple__healthy', 'Blueberry__healthy',
               'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)__Common_rust', 'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)__healthy', 'Grape__Black_rot',
               'Grape__Esca(Black_Measles)',
               'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange__Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach__healthy', 'Pepper,_bell__Bacterial_spot',
               'Pepper,bell_healthy', 'Potato_Early_blight', 'Potato__Late_blight',
               'Potato__healthy', 'Raspberry_healthy', 'Soybean__healthy',
               'Squash__Powdery_mildew', 'Strawberry__Leaf_scorch',
               'Strawberry__healthy', 'Tomato_Bacterial_spot', 'Tomato__Early_blight',
               'Tomato__Late_blight', 'Tomato_Leaf_Mold', 'Tomato__Septoria_leaf_spot',
               'Tomato__Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
               'Tomato___healthy']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in right shape
# def read_image(filename):
#     img = fk.load_img(filename, target_size=(224, 224))
#     x = fk.image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = fk.preprocess_input(x)
#     return x


def load_and_prep_image(filename, img_shape=256):
    """
    Reads an image from filename, turns it into a tensor
    and reshapes it to (img_shape, img_shape, colour_channel).
    """
    # Read in target file (an image)
    img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=3)

    # Resize the image (to the same size our model was trained on)
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img


@app.route('/predict', methods=['GET', 'POST'])
def pred_and_plot():
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    if fk.request.method == 'POST':
        file = fk.request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('images', filename)
            file.save(file_path)
            # Import the target image and preprocess it
            img = load_and_prep_image("images/"+ filename)
            # Make a prediction
            pred = model.predict(tf.expand_dims(img, axis=0))
            # Get the predicted class
            if len(pred[0]) > 1:  # check for multi-class
                # if more than one output, take the max
                pred_class = class_names[pred.argmax()]
            else:
                # if only one output, round
                pred_class = class_names[int(tf.round(pred)[0][0])]
            return fk.render_template('predict.html', prob=pred_class, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"

    # Plot the image and predicted class
    # plt.imshow(img)
    # plt.title(f"Prediction: {pred_class}")
    # plt.axis(False)


# pred_and_plot(model, "images/predictdidsease.jpg", class_names)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
