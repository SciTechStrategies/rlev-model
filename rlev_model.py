import cPickle as pickle
import os
import gzip

import sklearn.linear_model.logistic  # required for loading pickled model

DATA_DIR = "data"


def unpickle(name):
    pkl_file = os.path.join(DATA_DIR, "{0}.pkl.gz".format(name))
    with gzip.GzipFile(pkl_file) as infh:
        return pickle.load(infh)


def get_model():
    """ Return the model object. """
    return unpickle('model')


def get_title_features():
    """ Return map from title word to feature id. """
    return unpickle('title_features')


def get_abstract_features():
    """ Return map from abstract word to feature id. """
    return unpickle('abstract_features')


print(get_model())
print(get_title_features())
print(get_abstract_features())
