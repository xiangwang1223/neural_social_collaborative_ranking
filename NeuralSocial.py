import os
from time import time
import argparse
import random
from numpy.linalg import inv
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import numpy as np
import scipy as sp
from numpy import inf
import scipy.sparse

from Metrics import evaluate_pre_rec_auc,evaluate_auc
import DataSet as load_data


def parase_args():
    parser = argparse.ArgumentParser(description="Run DeepFM.")
    parser.add_argument('--dataset', nargs='?', default='twitter',
                        help='Choose a dataset.')
    parser.add_argument('--total_epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--layer_unit', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--drop_out', nargs='?', default='[1]',
                        help='Keep probability (1-dropout) of each layer. 1: no dropout. ')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--social_rate', type=float, default=0.5,
                        help='Social rate.')
    parser.add_argument('--user_rate', type=float, default=0.,
                        help='Social rate.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha')
    parser.add_argument('--item_rate', type=float, default=0.,
                        help='Attribute rate')
    parser.add_argument('--attr_rate', type=float, default=0.,
                        help='Attribute rate')
    parser.add_argument('--pooling_type', nargs='?', default='avg_pooling',
                        help='Specify a pooling type (none, avg_pooling or bilinear_pooling).')
    parser.add_argument('--loss_type', nargs='?', default='graph_embedding',
                        help='Specify a loss type (graph_embedding or pairwise_compare).')
    parser.add_argument('--optimizer_type', nargs='?', default='adag',
                        help='Specify an optimizer type (adam, adag, gd, mom).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Display the epoch result.')
    parser.add_argument('--random_seed', type=int, default=7),
    parser.add_argument('--topK', nargs='?', default='[5,10,15,20,25]',)
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    return parser.parse_args()


class NeuralSocial(BaseEstimator, TransformerMixin):
    def __init__(self, dataset, n_user, n_attr, n_item, layer_unit, drop_out, pooling_type, loss_type,
                 optimizer_type, learning_rate, social_rate, user_rate, attr_rate, item_rate, alpha,
                 random_seed, batch_size, total_epoch, verbose, topK, activation_function):
        self.dataset = dataset

        self.n_user = n_user
        self.n_attr = n_attr
        self.n_item = n_item

        # the first layer is the embedding layer; the latent factor is equal to the layer_unit[0].
        self.latent_factor = layer_unit[0]
        self.layer_unit = layer_unit
        # the first drop out is the keeping probability of embedding layer.
        self.drop_out = drop_out

        self.pooling_type = pooling_type
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.social_rate = social_rate
        self.user_rate = user_rate
        self.attr_rate = attr_rate
        self.item_rate = item_rate
        self.alpha = alpha

        self.random_seed = random_seed
        self.batch_size = batch_size
        self.total_epoch = total_epoch
        self.verbose = verbose
        self.topK = topK

        self.activation_function  = activation_function

        # List of performance regarding precision, selection, and recall.
        self.train_pre, self.valid_pre, self.test_pre = [], [], []
        self.train_sel, self.valid_sel, self.test_sel = [], [], []
        self.train_rec, self.valid_rec, self.test_rec = [], [], []
        self.train_auc, self.valid_auc, self.test_auc = [], [], []

        self._init_graph()
        self._valid_pair_socre()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.look_up_ids = tf.placeholder(tf.int32, shape=[None])

            self.train_labels = tf.placeholder(tf.float32, shape=[None])
            # ******** INFORMATION DOMAIN ********
            # Input training data with (batch_size,) shape.
            self.user_ids = tf.placeholder(tf.int32, shape=[None])
            # initialization.
            # self.user_rep = tf.placeholder(tf.float32, shape=[self.n_user, self.latent_factor])
            self.pos_item_ids = tf.placeholder(tf.int32, shape=[None])
            self.neg_item_ids = tf.placeholder(tf.int32, shape=[None])
            # .... The attribute selection matrices.
            self.user_attr_sel = tf.placeholder(tf.float32, shape=[None, self.n_attr, self.latent_factor])
            self.pos_item_attr_sel = tf.placeholder(tf.float32, shape=[None, self.n_attr, self.latent_factor])
            self.neg_item_attr_sel = tf.placeholder(tf.float32, shape=[None, self.n_attr, self.latent_factor])

            # ******** SOCIAL NETWORK *******
            # Input training data with (batch_size,) shape.
            # self.follower_ids = tf.placeholder(tf.int32, shape=[None])
            self.follower_rep = tf.placeholder(tf.float32, shape=[None, None])
            self.pos_fol_ids = tf.placeholder(tf.int32, shape=[None])
            self.neg_fol_ids = tf.placeholder(tf.int32, shape=[None])
            self.social_tie_weight = tf.placeholder(tf.float32, shape=[None])

            # ******** VALIDATION  *******
            # Input validation data.
            self.valid_user_ids = tf.placeholder(tf.int32, shape=[None])
            self.laplacian = tf.placeholder(tf.float32, shape=[self.n_user, self.latent_factor])
            self.valid_item_ids = tf.placeholder(tf.int32, shape=[None])
            self.valid_item_attr_sel = tf.placeholder(tf.float32, shape=[None, self.n_attr, self.latent_factor])

            self._init_weights()
            self._init_pooling()

            self._init_interaction_loss()
            # self._init_friendship_loss()
            # fetch the interaction user embedding to fix during training friendship.
            self.look_up_interaction_embedding = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.look_up_ids)

            # self._init_overall_loss()
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print "#params: %d" % total_parameters

    def _init_weights(self):
        self.weights = dict()

        self.weights['user_embeddings'] = tf.Variable(tf.random_normal(
            [self.n_user, self.latent_factor], 0., 0.1), name='user_embeddings')

        self.weights['item_embeddings'] = tf.Variable(tf.random_normal(
            [self.n_item, self.latent_factor], 0., 0.1), name='item_embeddings')

        self.weights['attr_embeddings'] = tf.Variable(tf.random_normal(
            [self.n_attr, self.latent_factor], 0., 0.1), name='attr_embeddings')

        for i in range(1, len(self.layer_unit)):
            if i == 0:
                pre_layer_unit = self.latent_factor
            else:
                pre_layer_unit = self.layer_unit[i - 1]

            glorot = np.sqrt(2. / (pre_layer_unit + self.layer_unit[i]))
            # ..Initialize weight parameters of hidden layers.
            self.weights['pair_layer_%d' % i] = tf.Variable(
                np.random.normal(loc=0., scale=glorot, size=(pre_layer_unit, self.layer_unit[i])), dtype=np.float32)
            self.weights['pair_bias_%d' % i] = tf.Variable(
                np.random.normal(loc=0., scale=glorot, size=(1, self.layer_unit[i])), dtype=np.float32)
        # ..prediction layer
        glorot = np.sqrt(2. / (self.layer_unit[-1] + 1))
        self.weights['prediction'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.layer_unit[-1], 1)), dtype=np.float32)
        # print 'All variables initialized done'

    def _init_pooling(self):
        # ________ INFORMATION DOMAIN __________
        user_rep = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_ids)
        pos_item_rep = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.pos_item_ids)
        neg_item_rep = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.neg_item_ids)

        user_attr_rep = tf.reshape(tf.mul(self.user_attr_sel, self.weights['attr_embeddings']),
                                   [-1, self.n_attr, self.latent_factor])
        pos_item_attr_rep = tf.reshape(tf.mul(self.pos_item_attr_sel, self.weights['attr_embeddings']),
                                       [-1, self.n_attr, self.latent_factor])
        neg_item_attr_rep = tf.reshape(tf.mul(self.neg_item_attr_sel, self.weights['attr_embeddings']),
                                       [-1, self.n_attr, self.latent_factor])

        # get the average pooling or bilinear pooling.
        flag = 'train'
        self.pos_pair_rep = self._pair_pooling(user_rep, user_attr_rep, pos_item_rep, pos_item_attr_rep, flag)
        self.neg_pair_rep = self._pair_pooling(user_rep, user_attr_rep, neg_item_rep, neg_item_attr_rep, flag)
        # drop out for the pooling layer to avoid overfitting.
        self.pos_pair_rep = tf.nn.dropout(self.pos_pair_rep, self.drop_out[0])
        self.neg_pair_rep = tf.nn.dropout(self.neg_pair_rep, self.drop_out[0])

        # ________ SOCIAL NETWORK DOMAIN __________
        # self.follower_rep = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.follower_ids)
        self.pos_fol_rep = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.pos_fol_ids)
        self.neg_fol_rep = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.neg_fol_ids)

    def _entity_pooling(self, entity_rep, entity_attr_rep):
        if self.pooling_type == 'avg_pooling':
            return tf.add(entity_rep, tf.reduce_mean(entity_attr_rep, 1))
        elif self.pooling_type == 'fm_pooling':
            return entity_rep
        elif self.pooling_type == 'bilinear_pooling':
            entity_sum_squared = tf.square(tf.add(entity_rep, tf.reduce_sum(entity_attr_rep, 1)))
            entity_squared_sum = tf.add(tf.square(entity_rep), tf.reduce_sum(tf.square(entity_attr_rep), 1))
            return 0.5 * tf.sub(entity_sum_squared, entity_squared_sum)

    def _pair_pooling(self, user_rep, user_attr_rep, item_rep, item_attr_rep, flag):
        if flag == 'train':
            user_rep = self._entity_pooling(user_rep, user_attr_rep)
        item_rep = self._entity_pooling(item_rep, item_attr_rep)

        pair_rep = tf.mul(user_rep, item_rep)

        return pair_rep


    # def _pair_pooling(self, user_rep, user_attr_rep, item_rep, item_attr_rep, flag):
    #     pair_rep = tf.add(user_rep, item_rep)
    #     if self.pooling_type == 'avg_pooling':
    #         # ________ AVERAGE __________
    #         if flag == 'train':
    #             user_rep = tf.add(user_rep, tf.reduce_mean(user_attr_rep, 1))
    #         item_rep = tf.add(item_rep, tf.reduce_mean(item_attr_rep, 1))

    #         pair_rep = tf.add(user_rep, item_rep)
    #     elif self.pooling_type == 'fm_pooling':
    #         sum_user_item_square = tf.square(tf.add(user_rep, item_rep))
    #         square_user_item_sum = tf.add(tf.square(user_rep), tf.square(item_rep))

    #         pair_rep = 0.5 * tf.sub(sum_user_item_square, square_user_item_sum)

    #     elif self.pooling_type == 'bilinear_pooling':
    #         # ________ FM __________
    #         # get the squared summed embeddings
    #         # ....get the sum of user and its attribute embeddings;
    #         user_rep_sum_squared = user_rep
    #         if flag == 'train':
    #             user_rep_sum_squared = tf.add(user_rep_sum_squared, tf.reduce_sum(user_attr_rep, 1))
    #         item_rep_sum_squared = tf.add(item_rep, tf.reduce_sum(item_attr_rep, 1))
    #         pair_sum_squared = tf.square(tf.add(user_rep_sum_squared, item_rep_sum_squared))

    #         # get the summed squared embeddings
    #         # ....get the squared of each embeddings;
    #         user_rep_squared_sum = tf.square(user_rep)
    #         if flag == 'train':
    #             user_rep_squared_sum = tf.add(user_rep_squared_sum, tf.reduce_sum(tf.square(user_attr_rep), 1))
    #         item_rep_squared_sum = tf.add(tf.square(item_rep), tf.reduce_sum(tf.square(item_attr_rep), 1))
    #         pair_squared_sum = tf.add(user_rep_squared_sum, item_rep_squared_sum)

    #         pair_rep = 0.5 * tf.sub(pair_sum_squared, pair_squared_sum)
    #     return pair_rep

    # Initiliaze the loss function of interaciton in information domain.
    def _init_interaction_loss(self):
        for i in range(1, len(self.layer_unit)):

            self.pos_pair_rep = tf.add(tf.matmul(self.pos_pair_rep, self.weights['pair_layer_%d' % i]),
                        self.weights['pair_bias_%d' % i])
            self.pos_pair_rep = self.activation_function(self.pos_pair_rep)
            self.pos_pair_rep = tf.nn.dropout(self.pos_pair_rep, self.drop_out[i])

            self.neg_pair_rep = tf.add(tf.matmul(self.neg_pair_rep, self.weights['pair_layer_%d' % i]),
                        self.weights['pair_bias_%d' % i])
            self.neg_pair_rep = self.activation_function(self.neg_pair_rep)
            self.neg_pair_rep = tf.nn.dropout(self.neg_pair_rep, self.drop_out[i])

        self.pos_pair_rep = tf.reduce_sum(tf.matmul(self.pos_pair_rep, self.weights['prediction']), 1)
        self.neg_pair_rep = tf.reduce_sum(tf.matmul(self.neg_pair_rep, self.weights['prediction']), 1)
        # Compute the loss.
        self.interaction_loss = tf.reduce_sum(tf.nn.l2_loss(self.pos_pair_rep - self.neg_pair_rep - self.train_labels))
        # self.interaction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #         tf.subtract(self.pos_pair_rep, self.neg_pair_rep), self.train_labels))
        # Optimizer.
        if self.optimizer_type == 'adam':
            self.optimizer_interaction = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.interaction_loss)
        elif self.optimizer_type == 'adag':
            self.optimizer_interaction = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.interaction_loss)
        elif self.optimizer_type == 'gd':
            self.optimizer_interaction = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.interaction_loss)
        elif self.optimizer_type == 'mom':
            self.optimizer_interaction = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.interaction_loss)

    # Initialize the loss function of interaction in social domain.
    def _init_friendship_loss(self):
        self.pos_friendship_rep = tf.reduce_sum(tf.mul(self.follower_rep, self.pos_fol_rep), 1)
        self.neg_friendship_rep = tf.reduce_sum(tf.mul(self.follower_rep, self.neg_fol_rep), 1)

        self.labels_concat = tf.concat(0, [self.train_labels, self.train_labels-self.train_labels])

        if self.loss_type == 'graph_embedding':
            # self.friendship_concat = tf.concat(0, [self.pos_friendship_rep, self.neg_friendship_rep])
            # self.weights_concat = tf.concat(0, [self.social_tie_weight, self.social_tie_weight])
            # self.friendship_loss = tf.reduce_sum(
            #     tf.mul(self.weights_concat, tf.nn.sigmoid_cross_entropy_with_logits(
            #         self.friendship_concat, self.labels_concat)))

            self.friendship_loss = tf.reduce_sum(
                tf.mul(self.social_tie_weight, tf.nn.sigmoid_cross_entropy_with_logits(
                    self.pos_friendship_rep, self.train_labels)))

        elif self.loss_type == 'l2_loss':
            self.friendship_loss = tf.reduce_sum(tf.nn.l2_loss(tf.sub(self.follower_rep, self.pos_fol_rep)))
        elif self.loss_type == 'l2_loss_reg':
            self.friendship_loss = tf.reduce_sum(tf.nn.l2_loss(tf.sub(self.follower_rep, self.pos_fol_rep)))
        elif self.loss_type == 'pairwise':
            self.friendship_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    tf.sub(self.pos_friendship_rep, self.neg_friendship_rep), self.train_labels))

        # Optimizer.
        if self.optimizer_type == 'adam':
            self.optimizer_friendship = tf.train.AdamOptimizer(learning_rate=self.social_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.friendship_loss)
        elif self.optimizer_type == 'adag':
            self.optimizer_friendship = tf.train.AdagradOptimizer(learning_rate=self.social_rate, initial_accumulator_value=1e-8).minimize(self.friendship_loss)
        elif self.optimizer_type == 'gd':
            self.optimizer_friendship = tf.train.GradientDescentOptimizer(learning_rate=self.social_rate).minimize(self.friendship_loss)
        elif self.optimizer_type == 'mom':
            self.optimizer_friendship = tf.train.MomentumOptimizer(learning_rate=self.social_rate, momentum=0.95).minimize(self.friendship_loss)

    # Initialize the loss function of all the losses.
    def _init_overall_loss(self):
        self.overall_loss = self.interaction_loss + self.social_rate * self.friendship_loss
        if self.user_rate > 0:
            self.overall_loss += tf.contrib.layers.l2_regularizer(self.user_rate)(self.weights['user_embeddings'])
        if self.item_rate > 0:
            self.overall_loss += tf.contrib.layers.l2_regularizer(self.item_rate)(self.weights['item_embeddings'])
        if self.attr_rate > 0:
            self.overall_loss += tf.contrib.layers.l2_regularizer(self.attr_rate)(self.weights['attr_embeddings'])

        # Optimizer.
        if self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.overall_loss)
        elif self.optimizer_type == 'adag':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.overall_loss)
        elif self.optimizer_type == 'gd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.overall_loss)
        elif self.optimizer_type == 'mom':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.overall_loss)


    def _valid_pair_socre(self):
        flag = 'test'
        valid_user_rep = tf.nn.embedding_lookup(self.laplacian, self.valid_user_ids)
        valid_item_rep = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.valid_item_ids)
        valid_item_attr_rep = tf.reshape(tf.mul(self.valid_item_attr_sel, self.weights['attr_embeddings']),
                                              [-1, self.n_attr, self.latent_factor])
        self.valid_pair_rep = self._pair_pooling(valid_user_rep, None, valid_item_rep, valid_item_attr_rep, flag)
        for i in range(1, len(self.layer_unit)):
            self.valid_pair_rep = tf.add(tf.matmul(self.valid_pair_rep, self.weights['pair_layer_%d' % i]),
                        self.weights['pair_bias_%d' % i])
            self.valid_pair_rep = self.activation_function(self.valid_pair_rep)
        self.valid_score = tf.matmul(self.valid_pair_rep, self.weights['prediction'])

    def _batch_fit_interaction(self, batch_data):
        feed_dict = {self.train_labels: batch_data['train_labels'],
                     self.user_ids: batch_data['user_ids'],
                     self.pos_item_ids: batch_data['pos_item_ids'],
                     self.neg_item_ids: batch_data['neg_item_ids'],

                     self.user_attr_sel: batch_data['user_attr_sel'],
                     self.pos_item_attr_sel: batch_data['pos_item_attr_sel'],
                     self.neg_item_attr_sel: batch_data['neg_item_attr_sel']}

        interaction_loss, batch_out = \
            self.sess.run((self.interaction_loss, self.optimizer_interaction), feed_dict=feed_dict)
        return interaction_loss

    def _batch_fit_friendship(self, batch_data):
        feed_dict = {self.train_labels: batch_data['train_labels'],

                     self.follower_rep: batch_data['follower_rep'],
                     self.pos_fol_ids: batch_data['pos_fol_ids'],
                     self.neg_fol_ids: batch_data['neg_fol_ids'],
                     self.social_tie_weight: batch_data['social_tie_weight']}

        friendship_loss, batch_out = \
            self.sess.run((self.friendship_loss, self.optimizer_friendship), feed_dict=feed_dict)
        return friendship_loss

    def _batch_fit_interaction_embedding(self, look_up_ids):
        feed_dict = {self.look_up_ids: look_up_ids}
        look_up_interaction_embedding = self.sess.run(self.look_up_interaction_embedding, feed_dict=feed_dict)
        return look_up_interaction_embedding


    def _fetch_batch_interaction(self, train, valid, test):
        batch_data = dict()
        random_interaction_index = random.sample(range(len(train['interaction_list'])), self.batch_size)

        selected_interactions = [train['interaction_list'][i] for i in random_interaction_index]

        all_user_attrs = train['user_attr_mat']
        all_item_attrs = train['item_attr_mat']

        batch_data['train_labels'] = [1. for i in selected_interactions]
        batch_data['user_ids'] = [i[0] for i in selected_interactions]
        batch_data['pos_item_ids'] = [i[1] for i in selected_interactions]
        batch_data['neg_item_ids'] = list()
        for user in batch_data['user_ids']:
            while True:
                neg_item_id = random.randint(0, self.n_item-1)
                if neg_item_id not in train['interaction_dict'][user]:
                    batch_data['neg_item_ids'].append(neg_item_id)
                    break

        batch_data['user_attr_sel'] = np.reshape(all_user_attrs[batch_data['user_ids']],
                                                 (self.batch_size, self.n_attr, self.latent_factor))
        batch_data['pos_item_attr_sel'] = np.reshape(all_item_attrs[batch_data['pos_item_ids']],
                                                     (self.batch_size, self.n_attr, self.latent_factor))
        batch_data['neg_item_attr_sel'] = np.reshape(all_item_attrs[batch_data['neg_item_ids']],
                                                     (self.batch_size, self.n_attr, self.latent_factor))
        return batch_data

    def _fetch_batch_friendship(self, train, valid, test, interaction_embedding):
        batch_data = dict()

        # random_friendship_index = list()

        # while True:
        #     random_index = random.randint(0, len(train['friendship_list'])-1)
        #     follower = train['friendship_list'][random_index][0]
        #     count = 0
        #     if follower not in valid['testRatings'].keys() and follower not in test['testRatings'].keys():
        #         random_friendship_index.append(random_index)
        #         count += 1
        #         if count == self.batch_size:
        #             break

        random_friendship_index = random.sample(range(len(train['friendship_list'])), self.batch_size)

        selected_friendship = [train['friendship_list'][i] for i in random_friendship_index]

        batch_data['train_labels'] = [1. for i in selected_friendship]
        batch_data['follower_ids'] = [i[0] for i in selected_friendship]
        batch_data['follower_rep'] = interaction_embedding[batch_data['follower_ids'],]
        batch_data['pos_fol_ids'] = [i[1] for i in selected_friendship]
        batch_data['neg_fol_ids'] = list()
        for follower in batch_data['follower_ids']:
            while True:
                neg_fol_id = random.randint(0, self.n_user-1)
                if neg_fol_id not in train['friendship_dict'][follower] and neg_fol_id not in train['interaction_dict'].keys():
                    batch_data['neg_fol_ids'].append(neg_fol_id)
                    break

        batch_data['social_tie_weight'] = [1./len(train['friendship_dict'][i]) for i in batch_data['follower_ids']]

        return batch_data

    def _fetch_valid_data(self, overall_data, current_user_id, laplacian):
        valid_data = dict()
        all_item_attrs = overall_data['item_attr_mat']

        valid_data['valid_user_ids'] = [current_user_id for i in range(self.n_item)]
        valid_data['valid_item_ids'] = [i for i in range(self.n_item)]
        valid_data['valid_item_attr_sel'] = np.reshape(all_item_attrs[valid_data['valid_item_ids']],
                                                       (-1, self.n_attr, self.latent_factor))
        valid_data['laplacian'] = laplacian
        return valid_data

    def _valid_fit(self, valid_dict):
        feed_dict = {self.valid_user_ids: valid_dict['valid_user_ids'],
                     self.valid_item_ids: valid_dict['valid_item_ids'],
                     self.valid_item_attr_sel: valid_dict['valid_item_attr_sel'],
                     self.laplacian: valid_dict['laplacian']}
        prediction_score = self.sess.run(self.valid_score, feed_dict=feed_dict)
        return prediction_score

    def _laplacian(self, train, valid, test):

        W = np.zeros(shape=[self.n_user, self.n_user])
        I = np.identity(self.n_user)

        for user in train['friendship_dict'].keys():
            friends = train['friendship_dict'][user]
            W[user, friends] = 1
            W[friends, user] = 1

        # print np.diag(np.power(W.sum(axis=1), -0.5))
        D_inv = np.diag(np.power(W.sum(axis=1), -0.5))
        D_inv[D_inv == inf] = 0

        D_inv = scipy.sparse.csc_matrix(D_inv)
        W = scipy.sparse.csc_matrix(W)

        S = D_inv * W * D_inv
        S = S.todense()

        L = inv(I - self.alpha * S)
        return L


    def _train(self, train, valid, test):
        # Training cycle.
        best_auc, corr_auc = 0., 0.
        best_pre_valid, best_sel_valid, best_rec_valid = [0. for k in range(len(self.topK))], [0. for k in range(len(self.topK))], [0. for k in range(len(self.topK))]
        corr_pre_test, corr_sel_test, corr_rec_test = [0. for k in range(len(self.topK))], [0. for k in range(len(self.topK))], [0. for k in range(len(self.topK))]

        L = self._laplacian(train, valid, test)

        for epoch in range(self.total_epoch):
            t1 = time()
            avg_cost = 0.
            avg_interaction_cost = 0.
            avg_friendship_cost = 0.

            total_batch = max(int(len(train['interaction_list']) / self.batch_size),
                              int(len(train['friendship_list'])) / self.batch_size)
            # Look up all batches.
            for iter in range(1):
                for i in range(total_batch):
                    # Fetch a batch data.
                    batch_data = self._fetch_batch_interaction(train, valid, test)

                    # Fit training using the batch data.
                    i_cost = self._batch_fit_interaction(batch_data)

                    # Compute average loss.
                    avg_cost += i_cost / total_batch
                    avg_interaction_cost += i_cost / total_batch

                # Init performance
                # prediction_score = np.asmatrix(np.zeros(shape=(self.n_user, self.n_item)))
                # for i in list(valid['testIds'] | test['testIds']):
                #     valid_dict = self._fetch_valid_data(train, i)
                #     current_score = self._valid_fit(valid_dict)
                #     prediction_score[i, :] = np.transpose(current_score)
                # auc_valid, pre_vec_valid, sel_vec_valid, rec_vec_valid = evaluate_pre_rec_auc(all_prediction_score=prediction_score,
                #                                                            testRatings=valid['testRatings'],
                #                                                            Ks=self.topK)
                # print 'Training@%d: AUC: %.4f; P@10: %.4f; S@10: %.4f; R@10: %.4f' \
                #               % (iter, auc_valid, pre_vec_valid[0], sel_vec_valid[0], rec_vec_valid[0])


            all_user_ids = range(self.n_user)
            all_user_interaction_embeddings = self._batch_fit_interaction_embedding(all_user_ids)

            if self.alpha == 0:
                L_pred = all_user_interaction_embeddings
            else:
                L_pred = L * all_user_interaction_embeddings



            # for iter in range(10):
            #     iter_cost = 0.
            #     for i in range(total_batch):
            #         # Fetch a batch data.
            #         batch_data = self._fetch_batch_friendship(train, valid, test, all_user_interaction_embeddings)

            #         # Fit training using the batch data.
            #         f_cost = self._batch_fit_friendship(batch_data)

            #         # Compute average loss.
            #         iter_cost += f_cost /total_batch
            #         avg_cost += f_cost / total_batch
            #         avg_friendship_cost += f_cost / total_batch

            #     print iter_cost

            #     # Init performance
            #     prediction_score = np.asmatrix(np.zeros(shape=(self.n_user, self.n_item)))
            #     for i in list(valid['testIds'] | test['testIds']):
            #         valid_dict = self._fetch_valid_data(train, i)
            #         current_score = self._valid_fit(valid_dict)
            #         prediction_score[i, :] = np.transpose(current_score)
            #     auc_valid, pre_vec_valid, sel_vec_valid, rec_vec_valid = evaluate_pre_rec_auc(all_prediction_score=prediction_score,
            #                                                                    testRatings=valid['testRatings'],
            #                                                                    Ks=self.topK)
            #     print 'Training@%d: AUC: %.4f; P@10: %.4f; S@10: %.4f; R@10: %.4f' \
            #                   % (iter, auc_valid, pre_vec_valid[0], sel_vec_valid[0], rec_vec_valid[0])

                          

            t2 = time()

            # prediction_score = np.asmatrix(np.zeros(shape=(self.n_user, self.n_item)))
            # for i in list(set(range(self.n_user)) - (valid['testIds'] | test['testIds'])):
            #     valid_dict = self._fetch_valid_data(train, i)
            #     current_score = self._valid_fit(valid_dict)
            #     prediction_score[i, :] = np.transpose(current_score)

            # auc_train, pre_vec_train, sel_vec_train, rec_vec_train = evaluate_pre_rec_auc(all_prediction_score=prediction_score,
            #                                                                 testRatings=train['interaction_dict'], Ks=self.topK)

            prediction_score = np.asmatrix(np.zeros(shape=(self.n_user, self.n_item)))
            for i in list(valid['testIds'] | test['testIds']):
                valid_dict = self._fetch_valid_data(train, i, L_pred)
                current_score = self._valid_fit(valid_dict)
                prediction_score[i, :] = np.transpose(current_score)

            auc_valid = evaluate_auc(all_prediction_score=prediction_score, testRatings=valid['testRatings'], elimateRatings=valid['elimateRatings'])
            auc_test = evaluate_auc(all_prediction_score=prediction_score, testRatings=test['testRatings'], elimateRatings=test['elimateRatings'])
            pre_vec_valid, sel_vec_valid, rec_vec_valid = evaluate_pre_rec_auc(all_prediction_score=prediction_score,
                                                                           testRatings=valid['testRatings'],
                                                                           Ks=self.topK)

            pre_vec_test, sel_vec_test, rec_vec_test = evaluate_pre_rec_auc(all_prediction_score=prediction_score,
                                                                        testRatings=test['testRatings'], Ks=self.topK)

            self.valid_pre.append(pre_vec_valid)
            self.valid_sel.append(sel_vec_valid)
            self.valid_rec.append(rec_vec_valid)
            self.valid_auc.append(auc_valid)

            self.test_pre.append(pre_vec_test)
            self.test_sel.append(sel_vec_test)
            self.test_rec.append(rec_vec_test)
            self.test_auc.append(auc_test)

            for k in range(len(self.topK)):
                if auc_valid > best_auc:
                    best_auc = auc_valid
                    corr_auc = auc_test

                if pre_vec_valid[k] > best_pre_valid[k]:
                    best_pre_valid[k] = pre_vec_valid[k]
                    corr_pre_test[k] = pre_vec_test[k]

                if sel_vec_valid[k] > best_sel_valid[k]:
                    best_sel_valid[k] = sel_vec_valid[k]
                    corr_sel_test[k] = sel_vec_test[k]

                if rec_vec_valid[k] > best_rec_valid[k]:
                    best_rec_valid[k] = rec_vec_valid[k]
                    corr_rec_test[k] = rec_vec_test[k]

            if (epoch + 1) % self.verbose == 0:
                # print '\tEpoch: %04d; Time: %.4f Loss: %.4f = %.4f + %.4f; AUC: %.4f; P@10: %.4f; S@10: %.4f; R@10: %.4f' \
                #       % (epoch + 1, t2 - t1, avg_cost, avg_interaction_cost, avg_friendship_cost, auc_train,
                #          pre_vec_train[0], sel_vec_train[0], rec_vec_train[0])
                # print '\tEpoch: %04d; Time: %.4f Loss: %.4f = %.4f + %.4f; AUC: %.4f; P@10: %.4f; S@10: %.4f; R@10: %.4f' \
                #       % (epoch + 1, t2 - t1, avg_cost, avg_interaction_cost, avg_friendship_cost, auc_valid,
                #          pre_vec_valid[0], sel_vec_valid[0], rec_vec_valid[0])
                print '\tEpoch: %04d; Time: %.4f Loss: %.4f = %.4f + %.4f; AUC: %.4f; P@5 = %.4f; R@5 = %.4f' \
                      % (epoch + 1, t2 - t1, avg_cost, avg_interaction_cost, avg_friendship_cost, auc_valid, pre_vec_valid[0], rec_vec_valid[0])

        # print '\tEnd. Best AUC = %.4f, P@5 = %.4f, S@5 = %.4f, R@5 = %.4f' % (
        # best_auc, best_pre_valid[0], best_sel_valid[0], best_rec_valid[0])
        # print '\t     Test AUC = %.4f, P@5 = %.4f, S@5 = %.4f, R@5 = %.4f' % (
        # corr_auc, corr_pre_test[0], corr_sel_test[0], corr_rec_test[0])

        # return best_auc, best_pre_valid, best_sel_valid, best_rec_valid, corr_pre_test, corr_sel_test, corr_rec_test
        print '\tEnd. Best AUC = %.4f; P@5 = %.4f; R@5 = %.4f' % (best_auc, best_pre_valid[0], best_rec_valid[0])
        print '\t     Test AUC = %.4f; P@5 = %.4f; R@5 = %.4f' % (corr_auc, corr_pre_test[0], corr_rec_test[0])

        return best_auc, corr_auc

if __name__ == '__main__':
    args = parase_args()

    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function == tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    file_name = "output/%s/loss=%s layers=%s drop_out=%s lr=%f sr=%f" \
                % (args.dataset, args.loss_type, args.layer_unit, args.drop_out, args.learning_rate, args.social_rate)

    file_hash = str(file_name) + '.output'
    file_output = open(file_hash, 'a')

    data = load_data.Dataset(path=args.dataset + '/', latent_factor=eval(args.layer_unit)[0])
    t1 = time()

    run_label = "OursFinal: dataset=%s, activation_function=%s drop_out=%s, layers=%s, pooling_type=%s, loss_type=%s," \
                "total_epoch=%d, batch_size=%d, learn_rating=%.4f, social_rating=%.4f, user_rate=%.4f, item_rate=%.4f," \
                "attr_rate=%.4f" \
                % (args.dataset, args.activation, args.drop_out,
                   args.layer_unit, args.pooling_type, args.loss_type, args.total_epoch,
                   args.batch_size, args.learning_rate, args.social_rate, args.user_rate, args.item_rate,
                   args.attr_rate)
    # print(run_label)
    file_output.write(run_label + '\n')

    neural_social = NeuralSocial(args.dataset, data.n_user, data.n_attr, data.n_item,
                                 eval(args.layer_unit), eval(args.drop_out), args.pooling_type, args.loss_type,
                                 args.optimizer_type, args.learning_rate, args.social_rate,
                                 args.user_rate, args.attr_rate, args.item_rate, args.alpha,
                                 args.random_seed, args.batch_size, args.total_epoch, args.verbose, eval(args.topK), activation_function)

    neural_social._train(train=data.train, valid=data.valid, test=data.test)

    # best_valid_pre = np.max(np.asmatrix(neural_social.valid_pre), 0)
    # best_valid_sel = np.max(np.asmatrix(neural_social.valid_sel), 0)
    # best_valid_rec = np.max(np.asmatrix(neural_social.valid_rec), 0)

    # best_test_pre = np.max(np.asmatrix(neural_social.test_pre), 0)
    # best_test_sel = np.max(np.asmatrix(neural_social.test_sel), 0)
    # best_test_rec = np.max(np.asmatrix(neural_social.test_rec), 0)

    best_valid_auc = np.max(neural_social.valid_auc)
    best_test_auc = np.max(neural_social.test_auc)

    
    # for k in range(len(neural_social.topK)):
    outstream = 'Best Iter = %d\tAUC = %.4f' % (neural_social.valid_auc.index(best_valid_auc),best_valid_auc)
    # outstream = 'P@%d = %.4f\tS@%d = %.4f\tR@%d = %.4f' % (k, best_valid_pre[0,k],
    #                                                        k, best_valid_sel[0,k],
    #                                                        k, best_valid_rec[0,k])
    print outstream
    file_output.write(outstream + '\n')

    # for k in range(len(neural_social.topK)):
    outstream = 'Best Iter = %d\tAUC = %.4f\t' % (neural_social.test_auc.index(best_test_auc),best_test_auc)
    # outstream = 'P@%d = %.4f\tS@%d = %.4f\tR@%d = %.4f' % (k, best_test_pre[0,k],
    #                                                        k, best_test_sel[0,k],
    #                                                        k, best_test_rec[0,k])
    print outstream
    file_output.write(outstream + '\n')
    file_output.write(outstream + '\n')
    file_output.close()