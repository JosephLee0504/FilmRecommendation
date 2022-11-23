import tensorflow as tf
import pickle
from dataset import Dataset, decompression_feature
from features import _get_user_feature, _get_movie_feature
from inference import movie_feature_network, user_feature_network
import numpy as np


class FM():
    def __init__(self, k, train=True, y_testmode='regression'):
        self.k = k
        self.mode = y_testmode
        self.train = train
        self.save_dir = 'data/fmmodel/model'
        self.restore_dir = 'data/fmmodel'
        if not train:
            self.build_model(800167, 400)
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fm' in v.name])
            sess.run(tf.global_variables_initializer())
            cpkt = tf.train.get_checkpoint_state(self.restore_dir)
            saver.restore(sess, cpkt.model_checkpoint_path)

    def load_data(self):
        pass

    def build_model(self, n, p):
        self.n, self.p = n, p

        # design matrix
        with tf.name_scope('fm'):
            self.user = tf.placeholder('float', shape=[None, self.p / 2])
            self.movie = tf.placeholder('float', shape=[None, self.p / 2])
            self.X = tf.concat([self.user, self.movie], 1)
            # target vector
            self.y = tf.placeholder('float', shape=[None, 1])

            # bias and weights
            w0 = tf.Variable(tf.zeros([1]))
            W = tf.Variable(tf.zeros([self.p]))

            # interaction factors, randomly initialized
            V = tf.Variable(tf.random.normal([self.k, self.p], stddev=0.01))

            # estimator of y, initialized to 0
            y_hat = tf.Variable(tf.zeros([self.n, 1]))

            linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, self.X), 1, keepdims=True))
            pair_interaction = (tf.multiply(0.5,
                                            tf.reduce_sum(
                                                tf.subtract(
                                                    tf.pow(tf.matmul(self.X, tf.transpose(V)), 2),
                                                    tf.matmul(tf.pow(self.X, 2), tf.transpose(tf.pow(V, 2)))),
                                                1, keepdims=True)))
            self.y_hat = tf.add(linear_terms, pair_interaction)

            # lambda_w = tf.constant(0.001, name='lambda_w')
            # lambda_v = tf.constant(0.001, name='lambda_v')
            lambda_w = tf.constant(0.00, name='lambda_w')
            lambda_v = tf.constant(0.00, name='lambda_v')
            l2_norm = tf.add(
                tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W, 2))),
                tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V, 2)))
            )

            if self.mode == 'regression' and self.train:
                self.error = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_hat)))
                self.loss = tf.add(self.error, l2_norm)
            elif self.mode == 'classification' and self.train:
                print(self.y.get_shape().as_list())
                print(self.y_hat.get_shape().as_list())
                self.error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_hat))
                self.loss = tf.add(self.error, l2_norm)
                print(self.loss.get_shape().as_list())
                print(l2_norm.get_shape().as_list())

            # self.optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.5).minimize(self.loss)
            if self.train:
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)

    def do_train(self, x, y):
        epochs = 5
        batch_size = 256
        init = tf.global_variables_initializer()
        # self.sess = tf.Session()
        # self.sess.run(init)
        batch_per_epcho = (self.n + batch_size - 1) // batch_size
        saver = tf.train.Saver([v for v in tf.trainable_variables() if 'fm' in v.name])
        sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            cnt = 0
            dataset = Dataset(x.values, y.values)
            for bath_i in range(batch_per_epcho):
                train_x, train_y = dataset.next_batch(batch_size)
                users, movies = decompression_feature(train_x)
                # import ipdb;ipdb.set_trace()
                user_features_train = _get_user_feature(users)
                movie_features_train = _get_movie_feature(movies)
                _, loss = sess.run((self.optimizer, self.loss), feed_dict={self.user: user_features_train,
                                                                           self.movie: movie_features_train,
                                                                           self.y: train_y})
                if cnt % 30 == 0:
                    print("Epoch: %d, Loss: %.3f" % (epoch + 1, loss))
                cnt += 1
        saver.save(sess, self.save_dir, global_step=epoch)

    def test(self, x):
        dataset = Dataset(x.values)
        test_x = dataset.next_batch(x.shape[0])
        users, movies = decompression_feature(test_x)
        # import ipdb;ipdb.set_trace()
        user_features_test = _get_user_feature(users)
        movie_features_test = _get_movie_feature(movies)
        y_hat = self.sess.run((self.y_hat), feed_dict={self.user: user_features_test,
                                                       self.movie: movie_features_test,
                                                       # self.y: test_y
                                                       })
        return y_hat

    def predict(self, user_feature, movie_feature):
        y_hat = self.sess.run((self.y_hat), feed_dict={self.user: user_feature,
                                                       self.movie: movie_feature,
                                                       # self.y: test_y
                                                       })
        return y_hat

    def evaluate(self, x, y):
        if self.mode == 'regression':
            errors = []
            dataset = Dataset(x.values, y.values)
            test_x, test_y = dataset.next_batch(x.shape[0])
            users, movies = decompression_feature(test_x)
            # import ipdb;ipdb.set_trace()
            user_features_test = _get_user_feature(users)
            movie_features_test = _get_movie_feature(movies)
            error, y_hat = self.sess.run((self.error, self.y_hat), feed_dict={self.user: user_features_test,
                                                                              self.movie: movie_features_test,
                                                                              self.y: test_y
                                                                              })
            errors.append(error)
            MSE = np.array(errors).mean()
            print("MSE: ", MSE)
        elif self.mode == 'classification':
            pass
            # pred = np.zeros((len(self.X_test), 1))
            # for batchX, batchY in batcher(self.X_test, self.y_test):
            #     logits = self.sess.run(self.y_hat, feed_dict={self.X: batchX.reshape(-1, self.p), self.y: batchY.reshape(-1, 1)})
            #     y_hat = sigmoid(logits)
            #     pred[np.where(y_hat > 0.5)] = 1
            #     pred[np.where(y_hat < 0.5)] = -1
            # print("Accuracy: ", np.mean(self.y_test == pred))
        self.sess.close()


if __name__ == '__main__':
    with open('./data/data.p', 'rb') as data:
        train_X, train_y, test_X, test_y = pickle.load(data)

    # train and evaluate
    # import ipdb;ipdb.set_trace()
    fm = FM(20)
    fm.build_model(800167, 400)
    fm.do_train(train_X, train_y)
    fm.evaluate(test_X, test_y)