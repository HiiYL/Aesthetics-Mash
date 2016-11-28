from flask import Flask, request, redirect, url_for, render_template,send_from_directory,session
from flask import g
from werkzeug import secure_filename

import os
import pandas as pd
from pandas import HDFStore
import numpy as np


from model import VGG_19_GAP_functional
from utils import preprocess_image,deprocess_image
import cv2

from flask_sqlalchemy import SQLAlchemy

from datetime import datetime

import tempfile


UPLOAD_FOLDER = 'uploads/'
HEATMAP_FOLDER = 'heatmaps/'
TEST_IMAGES_FOLDER='test_images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEST_IMAGES_FOLDER'] = TEST_IMAGES_FOLDER
app.config['HEATMAP_FOLDER'] = HEATMAP_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(
    tempfile.gettempdir(), 'test.db')

delta = 1.5

db = SQLAlchemy(app)

# To read from sqlite3 database
# import sqlite3
# import pandas as pd
# cnx = sqlite3.connect('/tmp/test.db')
# pd.read_sql('SELECT * from game_session',cnx)


# from main import db
# from main import GameSession
# db.drop_all()
# db.create_all()
# game_session = GameSession(24,16,18,delta)
# db.session.add(game_session)
# db.session.commit()
# GameSession.query.all()
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True)
#     email = db.Column(db.String(120), unique=True)

#     def __init__(self, username, email):
#         self.username = username
#         self.email = email

#     def __repr__(self):
#         return '<User %r>' % self.username

class GameSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    games_count = db.Column(db.Integer)
    player_win_count = db.Column(db.Integer)
    player_win_rate = db.Column(db.Float)
    ai_win_count = db.Column(db.Integer)
    ai_win_rate = db.Column(db.Float)
    delta = db.Column(db.Float)
    game_date = db.Column(db.DateTime)

    def __init__(self, games_count, player_win_count, ai_win_count,delta, game_date=None):
        self.games_count = games_count
        self.player_win_count = player_win_count
        self.player_win_rate = player_win_count / games_count
        self.ai_win_count = ai_win_count
        self.ai_win_rate = ai_win_count / games_count
        self.delta = delta
        if game_date is None:
            game_date = datetime.utcnow()
        self.game_date = game_date

    def __repr__(self):
        return '<Post {}> {} {} {} {} {}'.format(self.id, self.games_count,self.player_win_count, self.ai_win_count,self.ai_win_rate, self.delta)


model = VGG_19_GAP_functional("aesthestic_gap_weights_1.h5", heatmap=True)

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


ava_path = "dataset/AVA/data/"
store = HDFStore('dataset/labels.h5')
ava_table = store['labels_test']
@app.route('/', methods=['GET', 'POST'])
def root():
    base = ava_table.ix[np.random.choice(ava_table.index, 1)[0]]
    to_compare_df = ava_table.ix[abs(ava_table['score'] - base.score) > delta]
    to_compare = to_compare_df.ix[np.random.choice(to_compare_df.index, 1)[0]]

    return render_template('index.html', comparison_set=[base, to_compare])

@app.route("/stats")
def stats():
    total_session = GameSession.query.count()
    games_played_percentile = 100 * GameSession.query.filter(GameSession.games_count > session['total']).count() / total_session
    games_won_percentile = 100 * GameSession.query.filter(GameSession.player_win_count > session['current']).count() / total_session

    win_rate = session['current'] / session['total']
    games_winrate_percentile = 100 * GameSession.query.filter(GameSession.player_win_rate > win_rate).count() / total_session

    winrate_against_ai_percentile = 100 * GameSession.query.filter(GameSession.ai_win_rate > win_rate).count() / total_session
    return render_template('stats.html', args=[games_played_percentile,games_won_percentile,win_rate,games_winrate_percentile,winrate_against_ai_percentile])
@app.route("/reset")
def reset():
    game_session = GameSession(session['total'],session['current'],session['cpu_current'], delta)
    db.session.add(game_session)
    db.session.commit()
    session.clear()
    return redirect(url_for('root'))

@app.route('/compare', methods=['POST'])
def compare():
    values = request.values
    base = values['base']
    to_compare = values['compare']

    selected = int(values['selected'])

    base = ava_table.ix[int(base)]
    to_compare = ava_table.ix[int(to_compare)]

    winner = np.argmax([base.score,to_compare.score])

    print(winner)
    print(selected)

    correct = (winner == selected)

    print(correct)

    if session.get('current') and session.get('total'):
        if correct:
            session['current'] += 1
        session['total'] += 1
    else:
        session['current'] = 1 if correct else 0
        session['total'] = 1

    base_image_path = 'test_images/{}.jpg'.format(base.name)
    compare_image_path = 'test_images/{}.jpg'.format(to_compare.name)

    heights = []
    widths = []

    base_img_original = cv2.imread(base_image_path)
    width, height, _ = base_img_original.shape
    base_im = preprocess_image(base_img_original)


    widths.append(width)
    heights.append(height)

    compare_img_original = cv2.imread(compare_image_path)
    width, height, _ = compare_img_original.shape
    compare_im = preprocess_image(compare_img_original)

    widths.append(width)
    heights.append(height)

    stacked_im = np.vstack([base_im, compare_im])

    out = model.predict(stacked_im)

    scores = out[0]
    good_class_confidence = scores[:,1]
    cpu_winner_predict = np.argmax(good_class_confidence)
    cpu_correct = (winner == cpu_winner_predict)

    
    if session.get('cpu_current'):
        if cpu_correct:
            session['cpu_current'] += 1
    else:
        session['cpu_current'] = 1 if cpu_correct else 0

    conv_outputs = out[1]

    for i in range(2):
        conv_output = conv_outputs[i]
        if i == 0:
            process_and_generate_heatmap(conv_output, 
                model, heights[i], widths[i], base_img_original, base_im, '{}.jpg'.format(base.name))
        elif i == 1:
            process_and_generate_heatmap(conv_output, 
                model, heights[i], widths[i], compare_img_original, compare_im, '{}.jpg'.format(to_compare.name))

    return render_template('index.html', comparison_set=[base, to_compare],selected=selected,cpu_selected=cpu_winner_predict, groundtruth=winner, correct=correct, cpu_confidence=good_class_confidence)

def process_and_generate_heatmap(conv_output, model, height, width, img_original, img, img_name):
    cam = np.zeros(dtype = np.float32, shape = conv_output.shape[1:3])
    
    class_weights = model.layers[-1].get_weights()[0]

    class_to_visualize = 1 # 0 for bad, 1 for good
    for j, w in enumerate(class_weights[:, class_to_visualize]):
            cam += w * conv_output[j, :, :]
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap*0.5 + img_original
    # filename = '{}.jpg'.format(base.name)
    cv2.imwrite(os.path.join(app.config['HEATMAP_FOLDER'], img_name),img)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  return send_from_directory(app.config['UPLOAD_FOLDER'],
    filename)

@app.route('/heatmaps/<filename>')
def heatmap_file(filename):
  return send_from_directory(app.config['HEATMAP_FOLDER'],
    filename)

@app.route('/test_images/<filename>')
def test_image_file(filename):
  return send_from_directory(app.config['TEST_IMAGES_FOLDER'],
    filename)

if __name__ == "__main__":
    # app.debug = True
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    # print("YOUR IP ADDRESS IS: {0}".format(ni.ifaddresses('en0')[2][0]['addr']))
    app.run(host='0.0.0.0')
    # app.run()
