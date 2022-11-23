## 策略：
## 新人来了选 15个 评分 top的内容，剩下5个随机选

import pickle
import random
with open('./data/top_movies.p', 'rb') as top_movies_fs:
    top_movies = pickle.load(top_movies_fs)[:15]

with open('./data/other_movies.p', 'rb') as other_movies_fs:
    other_movies = pickle.load(other_movies_fs)

with open('./data/movies.p', 'rb') as movie:
    movies = pickle.load(movie)

last_movies = random.sample(other_movies, 5)
all_movies = top_movies + last_movies

# inplace algorithm 不需要赋值！！！！
random.shuffle(all_movies)
# import ipdb;ipdb.set_trace()

for movie in all_movies:
    if movie[0] >= len(movies):
        continue
    print(movies.loc[movie[0]])
    print('*****************************')