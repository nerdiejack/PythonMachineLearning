import os
import sqlite3

if os.path.exists('../data/reviews.sqlite'):
    os.remove('../data/reviews.sqlite')
conn = sqlite3.connect('../data/reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db'\
          ' (review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'I love this movie'
c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('now'))", (example1, 1))

example2 = 'I disliked this movie'
c.execute("INSERT INTO review_db"\
          " (review, sentiment, date) VALUES"\
          " (?, ?, DATETIME('now'))", (example2, 0))
conn.commit()
conn.close()

conn = sqlite3.connect('../data/reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date"\
    " BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')")
results = c.fetchall()

conn.close()
print(results)
