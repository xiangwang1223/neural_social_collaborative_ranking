import numpy as np


class Dataset(object):
    def __init__(self, path, latent_factor):

        self.n_attr = len(open(path + 'attr.list').readlines())
        self.n_item = len(open(path + 'item.attr').readlines())
        self.n_user = len(open(path + 'train.attr').readlines())

        self.latent_factor = latent_factor

        self.user_attr_mat = self._load_entity_attr(self.n_user, path + 'train.attr')
        self.item_attr_mat = self._load_entity_attr(self.n_item, path + 'item.attr')

        self.train = self._load_train(path + 'train.rating', path + 'train.social')

        self.valid = self._load_valid(path + 'valid.rating', [path +'elimate.rating', path + 'test.rating'])
        self.test = self._load_valid(path + 'test.rating', [path + 'elimate.rating', path + 'valid.rating'])

    def _load_entity_attr(self, n_entity, entity_attr_path):
        entity_attr_list = open(entity_attr_path, 'r').readlines()
        mat = np.zeros(shape=[n_entity, self.n_attr, self.latent_factor])

        for pairs in entity_attr_list:
            temps = pairs.split('\t')
            entity = int(temps[0].strip())
            attrs = [int(i.strip()) for i in temps[1:]]

            for attr in attrs:
                mat[entity, attr, ] = 1
        return mat

    def _load_train(self, interaction_path, friendship_path):
        train = dict()
        train['interaction_list'] = [(int(i.split('\t')[0].strip()), int(i.split('\t')[1].strip()))
                                     for i in open(interaction_path, 'r').readlines()]
        train['friendship_list'] = [(int(i.split('\t')[0].strip()), int(i.split('\t')[1].strip()))
                                    for i in open(friendship_path, 'r').readlines()]
        train['interaction_dict'] = self._pos_set(train['interaction_list'])
        train['friendship_dict'] = self._pos_set(train['friendship_list'])

        train['user_attr_mat'] = self.user_attr_mat
        train['item_attr_mat'] = self.item_attr_mat
        return train

    def _pos_set(self, pos_pair_list):
        pos_dict = dict()
        for pair in pos_pair_list:
            centorid = pair[0]
            pos_item = pair[1]

            if centorid in pos_dict.keys():
                pos_dict[centorid].append(pos_item)
            else:
                pos_dict[centorid] = [pos_item]
        return pos_dict

    def _load_valid(self, rating_path, elimate_paths=None):
        valid = dict()
        valid['testRatings'] = dict()
        valid['testIds'] = set()
        valid['elimateRatings'] = dict()

        valid_pairs = open(rating_path, 'r').readlines()
        for pair in valid_pairs:
            temps = pair.split('\t')
            centroid = int(temps[0].strip())
            valid['testIds'].add(centroid)
            valid['testRatings'][centroid] = [int(i.strip()) for i in temps[1:]]

        for elimate_path in elimate_paths:
            elimate_pairs = open(elimate_path, 'r').readlines()
            for pair in elimate_pairs:
                temps = pair.split('\t')
                centroid = int(temps[0].strip())
                if centroid not in valid['elimateRatings'].keys():
                    valid['elimateRatings'][centroid] = []
                for i in temps[1:]:
                    valid['elimateRatings'][centroid].append(int(i.strip()))

        return valid
