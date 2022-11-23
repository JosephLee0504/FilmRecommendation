import logging
import pickle

import numpy as np
import tensorflow as tf

from dataset import Dataset
from inference import full_network

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

BATCH_SIZE = 256
DROPOUT_PROB = 1


def main(model_path):
    user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_genres = tf.placeholder(tf.float32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    movie_title_length = tf.placeholder(tf.float32, [None], name='movie_title_length')
    dropout_keep_prob = tf.constant(DROPOUT_PROB, dtype=tf.float32, name='dropout_keep_prob')

    user_feature, movie_feature, _ = full_network(user_id, user_gender, user_age, user_job, movie_id,
                                                  movie_genres, movie_titles, movie_title_length,
                                                  dropout_keep_prob)

    with tf.variable_scope('user_movie_fc', reuse=True):
        user_movie_fc_kernel = tf.get_variable('kernel')
        user_movie_fc_bias = tf.get_variable('bias')

    with open('./data/users.p', 'rb') as users:
        user_Xs = pickle.load(users)
    with open('./data/movies.p', 'rb') as movies:
        movie_Xs = pickle.load(movies)

    user_dataset = Dataset(user_Xs.values, shuffle=False)
    movie_dataset = Dataset(movie_Xs.values, shuffle=False)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cpkt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, cpkt.model_checkpoint_path)

        # 根据之前的深度学习网络，提取用户特征
        user_features = {}
        user_features_np = None
        for batch in range((user_dataset.size + BATCH_SIZE - 1) // BATCH_SIZE):

            data = user_dataset.next_batch(BATCH_SIZE)
            feed = {
                user_id: np.reshape(data.take(0, 1), [len(data), 1]),
                user_gender: np.reshape(data.take(4, 1), [len(data), 1]),
                user_age: np.reshape(data.take(5, 1), [len(data), 1]),
                user_job: np.reshape(data.take(3, 1), [len(data), 1]),
            }
            feature = sess.run(user_feature, feed_dict=feed)
            if batch == 0:
                user_features_np = feature
            else:
                user_features_np  = np.concatenate((user_features_np, feature), axis=0)
            user_features.update({key: value for (key, value) in zip(data.take(0, 1), feature)})
        with open('./data/user-features-np.p', 'wb') as mf:
            pickle.dump(user_features_np, mf)
        with open('./data/user-features.p', 'wb') as uf:
            pickle.dump(user_features, uf)

        # 提取电影特征
        movie_features = {}
        movie_features_np = None
        # import ipdb;ipdb.set_trace()
        for batch in range((movie_dataset.size + BATCH_SIZE - 1) // BATCH_SIZE):
            data = movie_dataset.next_batch(BATCH_SIZE)
            feed = {
                movie_id: np.reshape(data.take(0, 1), [len(data), 1]),
                movie_genres: np.array(list(data.take(4, 1))),
                movie_titles: np.array(list(data.take(5, 1))),
                movie_title_length: (np.array(list(data.take(5, 1))) != 0).sum(axis=1)
            }
            feature = sess.run(movie_feature, feed_dict=feed)
            if batch == 0:
                movie_features_np = feature
            else:
                movie_features_np  = np.concatenate((movie_features_np, feature), axis=0)
            movie_features.update({key: value for (key, value) in zip(data.take(0, 1), feature)})
        with open('./data/movie-features-np.p', 'wb') as mf:
            pickle.dump(movie_features_np, mf)
        with open('./data/movie-features.p', 'wb') as mf:
            pickle.dump(movie_features, mf)

        # 保存最后损失层的kenel和bias为之后预测使用
        kernel, bais = sess.run([user_movie_fc_kernel, user_movie_fc_bias])
        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k,v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        # 这里的逻辑帮你理解模型结构
        # import ipdb;ipdb.set_trace()
        with open('./data/user-movie-fc-param.p', 'wb') as params:
            pickle.dump((kernel, bais), params)

modelobj = {}

def get_model():
    user_id = tf.placeholder(tf.int32, [None, 1], name='user_id')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')
    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_genres = tf.placeholder(tf.float32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, 15], name='movie_titles')
    movie_title_length = tf.placeholder(tf.float32, [None], name='movie_title_length')
    dropout_keep_prob = tf.constant(DROPOUT_PROB, dtype=tf.float32, name='dropout_keep_prob')

    if 'user_feature' not in modelobj:
        user_feature, movie_feature, _ = full_network(user_id, user_gender, user_age, user_job, movie_id,
                                                    movie_genres, movie_titles, movie_title_length,
                                                    dropout_keep_prob)
        modelobj['user_feature'] = user_feature
        modelobj['movie_feature'] = movie_feature
        modelobj['key'] = {
            'user_id': user_id,
            'user_gender': user_gender,
            'user_age': user_age,
            'user_job':user_job,
            'movie_id':movie_id,
            'movie_genres':movie_genres,
            'movie_titles':movie_titles,
            'movie_title_length': movie_title_length,
            'dropout_keep_prob': dropout_keep_prob}
        return user_feature, movie_feature, modelobj['key']
    else:
        return modelobj['user_feature'], modelobj['movie_feature'], modelobj['key']

user_feature, movie_feature, obj = get_model()
model_path='./data/model'
sess = tf.Session()
saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fm' not in v.name])
sess.run(tf.global_variables_initializer())
cpkt = tf.train.get_checkpoint_state(model_path)
#saver.restore(sess, cpkt.model_check_path)
def get_user_feature(user, model_path='./data/model'):
    # user [id(int), gender(0/1), age(0-6, jobid(int)]

    # user_feature, movie_feature, obj = get_model()
    user = np.array(user)
    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     sess.run(tf.global_variables_initializer())
    #     cpkt = tf.train.get_checkpoint_state(model_path)
    #     saver.restore(sess, cpkt.model_checkpoint_path)
        # import ipdb;ipdb.set_trace()
    feed = {
                obj['user_id']: np.reshape(user.take(0, 1), [len(user), 1]),
                obj['user_gender']: np.reshape(user.take(1, 1), [len(user), 1]),
                obj['user_age']: np.reshape(user.take(2, 1), [len(user), 1]),
                obj['user_job']: np.reshape(user.take(3, 1), [len(user), 1]),
            }
    feature = sess.run(user_feature, feed_dict=feed)
    return feature

def _get_user_feature(users, model_path='./data/model'):
    # user [id(int), gender(0/1), age(0-6, jobid(int)]

    # user_feature, movie_feature, obj = get_model()
    # import ipdb;ipdb.set_trace()
    # with tf.Session() as sess2:
        # sess2.run(tf.global_variables_initializer())
        # saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fm' not in v.name])
        # cpkt = tf.train.get_checkpoint_state(model_path)
        # saver.restore(sess2, cpkt.model_checkpoint_path)
        # import ipdb;ipdb.set_trace()
    feed = {
                obj['user_id']: users.id,
                obj['user_gender']: users.gender,
                obj['user_age']: users.age,
                obj['user_job']: users.job,
            }
    feature = sess.run(user_feature, feed_dict=feed)
    return feature

def to_onehot(index, length):
    z = np.zeros(length)
    z[index] = 1
    return z

def get_movie_feature(movie, model_path='./data/model'):
    # movie [id(int), movie_genres(0-17), movie_titles(int shape (n,15)), movie_title_length(int)]
    # genres {'Mystery': 0, 'Romance': 7, 'Sci-Fi': 2, "Children's": 8, 'Horror': 4, 'Film-Noir': 5, 'Crime': 6, 'Drama': 1, 'Fantasy': 3, 'Animation': 10, 'War': 15, 'Adventure': 11, 'Action': 12, 'Comedy': 13, 'Documentary': 14, 'Musical': 9, 'Thriller': 16, 'Western': 17}
    # user_feature, movie_feature, obj = get_model()

    # import ipdb;ipdb.set_trace()

    movie[0][1] = to_onehot([movie[0][1]], 18)
    movie = np.array(movie)
    # with tf.Session() as sess3:
        # sess3.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # cpkt = tf.train.get_checkpoint_state(model_path)
        # saver.restore(sess3, cpkt.model_checkpoint_path)
        # import ipdb;ipdb.set_trace()
    feed = {
            obj['movie_id']: np.reshape(movie.take(0, 1), [len(movie), 1]),
            obj['movie_genres']: np.array(list(movie.take(1, 1))),
            obj['movie_titles']: np.array(list(movie.take(2, 1))),
            obj['movie_title_length']: (np.array(list(movie.take(2, 1))) != 0).sum(axis=1)
        }
    feature = sess.run(movie_feature, feed_dict=feed)
    return feature

def _get_movie_feature(movies, model_path='./data/model'):
    # movie [id(int), movie_genres(0-17), movie_titles(int shape (n,15)), movie_title_length(int)]
    # genres {'Mystery': 0, 'Romance': 7, 'Sci-Fi': 2, "Children's": 8, 'Horror': 4, 'Film-Noir': 5, 'Crime': 6, 'Drama': 1, 'Fantasy': 3, 'Animation': 10, 'War': 15, 'Adventure': 11, 'Action': 12, 'Comedy': 13, 'Documentary': 14, 'Musical': 9, 'Thriller': 16, 'Western': 17}
    # user_feature, movie_feature, obj = get_model()
    # saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fm' not in v.name])
    # # import ipdb;ipdb.set_trace()

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     cpkt = tf.train.get_checkpoint_state(model_path)
    #     saver.restore(sess, cpkt.model_checkpoint_path)
        # import ipdb;ipdb.set_trace()
    feed = {
            obj['movie_id']: movies.id,
            obj['movie_genres']: movies.genres,
            obj['movie_titles']: movies.titles,
            obj['movie_title_length']: movies.title_length,
        }
    feature = sess.run(movie_feature, feed_dict=feed)
    return feature
if __name__ == '__main__':
    # 存储数据集里所有电影和用户的特征
    # main('./data/model')
    # 测试 单个 user 得到的特征
    print(get_user_feature([[3213, 1, 4, 32]]))
    # 测试 单个 movie 得到的特征
    print(get_movie_feature([[3212, 8, [231,421,4291,3038,1000,0,0,0,0,0,0,0,0,0,0]]]))
    # print(to_onehot([1,2,3], 18))