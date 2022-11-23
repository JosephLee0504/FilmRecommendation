## 策略：
## 老用户用户
## 召回策略：
## - 相似用户top 10打分最高的top 10 电影 100
## - 打分最高的电影  50
## - 随机25个电影
## - 个人 top 打分 的电影10 相似电影 10 最终 100个
## 排序：
## 使用fm模型和深度学习模型分别打分求和排序取top 20

import pickle
import random
import numpy as np
from utils import get_top_movies_by_user
from features import get_user_feature
from predict import _similar_user, predict_rating, relu, similar_movie
from fm import FM

fm = FM(20, False)

user = [[3213, 1, 4, 32]]
movies_ratings = [
    (380, 5),
    (391, 3),
    (380, 2),
    (1000, 3),
    (1001, 4),
]

user_feature = get_user_feature(user)
with open('./data/user-features.p', 'rb') as uf:
    user_features = pickle.load(uf, encoding='latin1')
with open('./data/movie-features.p', 'rb') as mf:
    movie_features = pickle.load(mf, encoding='latin1')

with open('./data/user-movie-fc-param.p', 'rb') as params:
    kernel, bais = pickle.load(params, encoding='latin1')

# user [id(int), gender(0/1), age(0-6, jobid(int)]

with open('./data/top_movies.p', 'rb') as top_movies_fs:
    # top 100个
    top_movies = pickle.load(top_movies_fs)[:100]

with open('./data/other_movies.p', 'rb') as other_movies_fs:
    other_movies = pickle.load(other_movies_fs)

with open('./data/movies.p', 'rb') as movie:
    movies_info = pickle.load(movie)

# 随机25个
last_movies = random.sample(other_movies, 25)
all_movies = top_movies + last_movies

# inplace algorithm 不需要赋值！！！！
random.shuffle(all_movies)

# 10个相似用户top 10 个电影
users = _similar_user(user_feature, 10, user_features)
for user in users:
    id = user[0]
    movies = get_top_movies_by_user(id, 10)
    all_movies += movies

# 10个高分电影的top 10 个相似电影
movies_ratings.sort(key=lambda x: x[1], reverse=True)
movies_ratings = movies_ratings[:10]
for movie, s in movies_ratings:
    if movie not in movie_features:
        continue
    s_movies = similar_movie(movie, 10, movie_features)
    all_movies += s_movies
all_movies = set(all_movies)

movie_score = []
# 排序， 给每个电影打分
for movie in all_movies:
    id = movie[0]
    if id >= len(movies_info):
        continue

    movie_feature = movie_features[id]
    user_f = user_feature.reshape(-1)

    s1 = predict_rating(user_f, movie_feature, kernel, bais, relu)[0]
    s2 = fm.predict(user_feature, movie_features[id].reshape(1, -1))[0][0]

    score = s1 + s2
    movie_score.append((id, score))

movie_score.sort(key=lambda x: x[1], reverse=True)
movie_scores = movie_score[:20]
# import ipdb;ipdb.set_trace()

for movie in movie_scores:
    if movie[0] >= len(movies_info):
        continue
    print(movies_info.loc[movie[0]])
    print('*****************************')
