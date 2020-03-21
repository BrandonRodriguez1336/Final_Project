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
from werkzeug import secure_filename
import logging

import os
import os

warnings.filterwarnings('ignore')


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["IMAGE_UPLOADS"] = "IMAGE_UPLOADS"

# session
# engine = create_engine(

#     "postgresql://test:test@localhost:5432/ProjectTwo", echo=False)


# Base = automap_base()
# Base.prepare(engine, reflect=True)
# Base.classes.keys()
# font_recognition = Base.classes.Font_recognition

# session = Session(engine)
# end session

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/upload-image", methods=["GET", "POST"])
# def upload_image():
#     if request.method == "POST":
#         if request.files:
#             image = request.files["image"]
#             image.save(os.path.join(
#                 app.config["IMAGE_UPLOADS"], image.filename))
#             print("Image saved")
#             return redirect("/")
def api_root():

    if request.method == 'POST' and request.files['image']:
        # app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']  # --->>> receive the image
        img_name = secure_filename(img.filename)
        # ----> we create the uploads folder
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        # app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)  # ----> saves the image inside the folder
        # score = pred_score  # ----> soon your machine learning score when u predict

        # get value
        # values = session.query(font_recognition.arial).all()
        # list = []
        # for value in values:
        #     dict_values = {"Font": value[0]}
        #     list.append(dict_values)
        pred_score = 0
        return "This is your predicted score " + str(pred_score)
    else:
        return "Where is the image?"


if __name__ == "__main__":
    app.run(debug=True)
