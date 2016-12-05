from main import db
from main import GameSession

db.drop_all()
db.create_all()


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