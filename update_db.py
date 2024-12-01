import sqlite3
import csv
import pickle
from vectorize import vectorize

dbname = 'Vector.db'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

def update_db(file:str):
    cur.execute(
    'CREATE TABLE IF NOT EXISTS embeddings(id INTEGER PRIMARY KEY AUTOINCREMENT,content STRING, vector BLOB, remarks STRING)')
    cur.execute('DELETE FROM embeddings')
    filepath = "DB_Vector/" + file
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "content":
                continue
            else:
                content = row[0]
                vector = vectorize(content)
                seriarized = pickle.dumps(vector)
                remarks = row[1]
                cur.execute('INSERT INTO embeddings (content,vector,remarks) VALUES (?,?,?)', (content,seriarized,remarks))
                conn.commit()
        conn.close()

update_db("vector_db_template.csv")