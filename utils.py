import pandas as pd
import pickle

def get_top_movies_by_user(userid, topk):
    with open('./data/ratings.p', 'rb') as meta:
        ratings = pickle.load(meta)
        row = ratings[ratings['UserID']==6040].sort_values(by=['ratings'], ascending=False)[:topk]
        movies = []
        for index, item in row.iterrows():
            movies.append((item['MovieID'], item['ratings']))
        return movies