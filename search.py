import sqlite3
import pickle
import numpy as np
from vectorize import vectorize

dbname = 'Vector.db'
conn = sqlite3.connect(dbname)
cur = conn.cursor()

def loadEmbeddings():
    embeddings = []
    cur.execute('SELECT * FROM embeddings')
    array = cur.fetchall()
    for item in array:
        np_array = pickle.loads(item[2])
        data = {
            "id": item[0],
            "content": item[1],
            "vector": np_array,
            "remarks": item[3]
        }
        embeddings.append(data)
    conn.close()
    return embeddings

def cos_sim(v1: np.array, v2: np.array):
    sim = np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    return sim

def search(input:str, count: int):
    array = []
    vector = vectorize(input)
    embeddings = loadEmbeddings()
    for embedding in embeddings:
        sim = cos_sim(vector, embedding["vector"])
        data = {
            "id": embedding["id"],
            "content": embedding["content"],
            "similarity": float(sim),
            "remarks": embedding["remarks"]
        }
        array.append(data)
    sorted_array = sorted(array,key=lambda x: x["similarity"], reverse=True)
    return sorted_array[:count]

#print(search("有害物質に関する表示", 5))