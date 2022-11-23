import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import tensorflow as tf

import os
import pickle
import re
from tensorflow.python.ops import math_ops
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile
import hashlib

class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def download_extract(database_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_ML1M = 'ml-1m'

    if database_name == DATASET_ML1M:
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        hash_code = 'c4d9eecfca2ab87c1945afe126590906'
        extract_path = os.path.join(data_path, 'ml-1m')
        save_path = os.path.join(data_path, 'ml-1m.zip')
        extract_fn = _unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    print('Done.')
    # Remove compressed data
#     os.remove(save_path)

def genres_multi_hot(genre_int_map):
    """
    电影类型使用multi-hot编码
    :param genre_int_map:genre到数字的映射字典
    :return:
    """

    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split('|')]
        multi_hot = np.zeros(len(genre_int_map))
        multi_hot[genre_int_list] = 1
        return multi_hot

    return helper

def title_encode(word_int_map):
    """
    将电影Title转成长度为15的数字列表，如果长度小于15则用0填充，大于15则截断
    :param word_int_map:word到数字的映射字段
    :return:
    """

    def helper(title):
        title_words = [word_int_map[word] for word in title.split()]
        if len(title_words) > 15:
            return np.array(title[:15])
        else:
            title_vector = np.zeros(15)
            title_vector[:len(title_words)] = title_words
            return title_vector

    return helper

def load_data():
    """
    Load Dataset from Zip File
    """
    # 读取User数据with open('data/ml-1m/users.dat') as users_raw_data:
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./data/ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')

    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['GenderIndex'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['AgeIndex'] = users['Age'].map(age_map)

    # 读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    # import ipdb; ipdb.set_trace()
    movies = pd.read_table('./data/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
    # 将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    movies['TitleWithoutYear'] = movies['Title'].map(lambda x: pattern.match(x).group(1))
    # 电影题材Multi-Hot编码
    genre_set = set()
    for val in movies['Genres'].str.split('|'):
        genre_set.update(val)

    genre_int_map = {val: ii for ii, val in enumerate(genre_set)}

    movies['GenresMultiHot'] = movies['Genres'].map(genres_multi_hot(genre_int_map))

    # 电影Title转数字列表,word的下标从1开始，0作为填充值
    word_set = set()
    for val in movies['TitleWithoutYear'].str.split():
        word_set.update(val)

    word_int_map = {val: ii for ii, val in enumerate(word_set, start=1)}

    movies['TitleIndex'] = movies['TitleWithoutYear'].map(title_encode(word_int_map))

    # 读取评分数据集
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./data/ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)
    # 存储 top 评分的 电影，跑过一遍之后 就可以注释非常慢，一般实际这种任务也是天级的。
    # save_top(data)

    # 将数据分成X和y两张表
    features, targets = data.drop(['ratings'], axis=1), data[['ratings']]

    return features, targets, age_map, gender_map, genre_int_map, word_int_map, users, movies, ratings

if __name__ == '__main__':
    data_dir = './data'
    download_extract('ml-1m', data_dir)

    users_title = ['UserID', 'Gender', 'Age', 'OccupationID', 'Zip-code']
    users = pd.read_table('./data/ml-1m/users.dat', sep='::', header=None, names=users_title, engine='python')
    print(users.head())

    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./data/ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine='python')
    print(movies.head())

    ratings_title = ['UserID', 'MovieID', 'Rating', 'timestamps']
    ratings = pd.read_table('./data/ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    print(ratings.head())

    features, targets, age_map, gender_map, genre_int_map, words_int_map, users, movies, ratings = load_data()

    with open('./data/meta.p', 'wb') as meta:
        pickle.dump((age_map, gender_map, genre_int_map, words_int_map), meta)

    with open('./data/users.p', 'wb') as meta:
        pickle.dump(users, meta)

    with open('./data/movies.p', 'wb') as meta:
        pickle.dump(movies, meta)

    with open('./data/ratings.p', 'wb') as meta:
        pickle.dump(ratings, meta)

    train_X, test_X, train_y, test_y = train_test_split(features, targets, test_size=0.2, random_state=0)
    with open('./data/data.p', 'wb') as data:
        pickle.dump((train_X, train_y, test_X, test_y), data)
    
    print(users.head())
    print(movies.head())
    print(ratings.head())