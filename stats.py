import sqlite3
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

# Create your connection.
cnx = sqlite3.connect('database/test.db')


df = pd.read_sql_query("SELECT * FROM game_session", cnx)


filtered_df = df[df['games_count'] > 5]

win_df = filtered_df[['player_win_count', 'ai_win_count','games_count']]

# filtered_df[['player_win_rate', 'ai_win_rate']].plot()
# plt.show()


player_win = len(df[df['player_win_count'] > df['ai_win_count'] ]) / len(df)

ai_win = len(df[df['ai_win_count'] > df['player_win_count'] ]) / len(df)

series = pd.Series([player_win,ai_win, 1 - ai_win - player_win ],
 index=['Player Victories', 'AI Victories', 'Draws'], name=" ")
series.plot.pie(figsize=(6, 6))

plt.show()