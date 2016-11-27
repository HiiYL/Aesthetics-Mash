from flask import Flask, request, redirect, url_for, render_template,send_from_directory,session
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


ava_path = "dataset/AVA/data/"
store = HDFStore('dataset/labels.h5')
ava_table = store['labels_test']
@app.route('/', methods=['GET', 'POST'])
def root():


    base = ava_table.ix[np.random.choice(ava_table.index, 1)[0]]
    to_compare_df = ava_table.ix[abs(ava_table['score'] - base.score) > 1.5]
    to_compare = to_compare_df.ix[np.random.choice(to_compare_df.index, 1)[0]]

    return render_template('index.html', comparison_set=[base, to_compare])

@app.route('/compare', methods=['POST'])
def compare():
    values = request.values
    base = values['base']
    to_compare = values['compare']

    to_compare_better = values['to_compare_better'] == 'true'

    base = ava_table.ix[int(base)]
    to_compare = ava_table.ix[int(to_compare)]

    winner = (to_compare.score > base.score)

    correct = (winner == to_compare_better)

    try:
        if correct:
          session['current'] += 1
        session['total'] += 1
    except KeyError:
        session['current'] = 1 if winner else 0
        session['total'] = 1

    ########
    #
    #  States
    #  0 - Both PC and CPU wrong
    #  1 - PC right, CPU wrong
    #  2 - PC wrong, CPU right
    #  3 - PC right, CPU right
    #
    ########

    return render_template('index.html', comparison_set=[base, to_compare], correct=correct)


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
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    # print("YOUR IP ADDRESS IS: {0}".format(ni.ifaddresses('en0')[2][0]['addr']))
    app.run()
    # app.run()
