import cPickle as pickle
import os
import gzip

import sklearn.linear_model.logistic  # required for loading pickled model

DATA_DIR = "data"


def get_model():
    """ Return the model object. """
    model_file = os.path.join(DATA_DIR, "model.pkl.gz")
    infh = gzip.GzipFile(model_file)
    return pickle.load(infh)


print(get_model())
