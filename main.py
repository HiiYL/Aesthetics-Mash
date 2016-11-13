from flask import Flask, request, redirect, url_for, render_template,send_from_directory
from flask import g
from werkzeug import secure_filename

import os
import pandas as pd
from pandas import HDFStore
import numpy as np

import netifaces as ni

UPLOAD_FOLDER = 'uploads/'
TEST_IMAGES_FOLDER='test_images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEST_IMAGES_FOLDER'] = TEST_IMAGES_FOLDER

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def root():
    ava_path = "dataset/AVA/data/"
    store = HDFStore('dataset/labels.h5')
    ava_table = store['labels_test']

    base = ava_table.ix[np.random.choice(ava_table.index, 1)]
    to_compare_df = ava_table.ix[abs(ava_table['score'] - base.score.values[0]) > 1.5]
    to_compare = to_compare_df.ix[np.random.choice(to_compare_df.index, 1)]

    return render_template('index.html', comparison_set=[base, to_compare])


@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

@app.route('/test_images/<filename>')
def test_image_file(filename):
  return send_from_directory(app.config['TEST_IMAGES_FOLDER'],
    filename)

if __name__ == "__main__":
    app.debug = True
    # print("YOUR IP ADDRESS IS: {0}".format(ni.ifaddresses('en0')[2][0]['addr']))
    app.run()
    # app.run()
