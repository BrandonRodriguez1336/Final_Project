import warnings
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.image import imread
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
from sqlalchemy import create_engine, func
from sqlalchemy.sql import text
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base
import sqlalchemy
import pandas as pd
import numpy as np
import datetime as dt
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory, request
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import logging

import os


warnings.filterwarnings('ignore')


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["IMAGE_UPLOADS"] = "IMAGE_UPLOADS"


app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/upload-image", methods=["GET", "POST"])
def api_root():

    if request.method == 'POST' and request.files['image']:
        # app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']  # --->>> receive the image
        img_name = secure_filename(img.filename)
        # ----> we create the uploads folder
        create_new_folder(app.config['UPLOAD_FOLDER'])
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        # app.logger.info("saving {}".format(image_path))
        img.save(image_path)  # ----> saves the image inside the folder
        # score = pred_score  # ----> soon your machine learning score when u predict

        # parse the image path inside tensors or rgb array. D:\\image.png ---> [[0,1,2], [2,1,3].....]
        # https://www.tensorflow.org/tutorials/load_data/images#performance
        
    
        image_path2 = 'C:\\Users\\ca25935\\Desktop\\UCD Data Analytics\\Final Project\\Final Project\\Final Project Repo\\Final_Project\\uploads\\'
        image_path2 = image_path2 + str(img_name)

        def load_image(img_path, show=False):

            img = image.load_img(img_path, target_size=(300, 300))
            img_tensor = image.img_to_array(img)                    # (height, width, channels)
            img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            return img_tensor

        new_image = load_image(image_path2)
    
       
        # prepare the image data (whatever you guys had on the ipynb) [[0,1,2], [2,1,3].....] ---> maybe you convert to 224 x 224
        #  
        #                                                                                      [  Arial, Sans, y, x]
        score = model.predict(new_image)
        # score = model.predict(image_path_in_array_format_rgb) #--> this should return a array of floats [ 0.2, 0.3, 0.4, 0.1]

        # return  y(our font predicted) --> highest probability that the model had predicted
        # include also the prediction score
        # font_predicted = y
        # prediction_score = max(score) * 100  # [ 0.2, 0.3, 0.4, 0.1] --> 40%

        return f"This is your predicted score 0  {score}"
        # return "This is your predicted score " + f"font predicted is {font_predicted} with confidence of {prediction_score}% "
    else:
        return "Where is the image?"


if __name__ == "__main__":
    # Load the h5 model that you have trained, before running the server
    model = load_model("Font_detector.h5")
    app.run(debug=True)
