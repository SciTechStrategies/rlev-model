"""SciTechStrategies Research Level Model

Usage:
  rlev_model.py <infile> [--encoding=<encoding>]

Options:
  -h --help              Show this screen
  --encoding=<encoding>  The encoding of the input text. Default is ISO-8859-2
"""

import codecs
import cPickle as pickle
import os
import gzip

from docopt import docopt
import numpy as np
from scipy.sparse import coo_matrix
import sklearn.linear_model.logistic  # required for loading pickled model

from word_feature_util import get_pid_abstr_title_features


DATA_DIR = "data"
ENCODING = "ISO-8859-2"
DELIMITER = u"\t"
MIN_N_FEAT = 2
MAX_N_FEAT = None


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


def csv_field_gen(infile, encoding=ENCODING):
    """ Generate fields from csv file. """
    for line in codecs.open(infile, 'r', encoding):
        yield line.rstrip().split(DELIMITER)


def get_id_features(infile, encoding=ENCODING):
    """ Returns a map from id to features.

    infile is a 3-column, delimited csv file
    first column is id
    second column is title
    third column is abstract
    """
    def id_abstr_gen():
        for fields in csv_field_gen(infile, encoding=encoding):
            if len(fields) == 3:
                yield fields[0], fields[2]

    def id_title_gen():
        for fields in csv_field_gen(infile, encoding=encoding):
            if len(fields) == 3:
                yield fields[0], fields[1]

    return get_pid_abstr_title_features(
        pid_abstr=id_abstr_gen(),
        abstr_features=get_abstract_features(),
        pid_title=id_title_gen(),
        title_features=get_title_features(),
    )


def get_n_features():
    """ Get total number of features. """
    return len(get_abstract_features()) + len(get_title_features())


def get_id_features_matrix(
    id_features,
    min_n_feat=MIN_N_FEAT,
    max_n_feat=MAX_N_FEAT,
):
    """
    Create a feature matrix, and also return row indexes of ids.
    """
    row = []
    col = []
    data = []
    ids = []
    i = 0
    for id, features in id_features:
        if min_n_feat is None or len(features) >= min_n_feat:
            if max_n_feat is None or len(features) <= max_n_feat:
                for word_index in features:
                    row.append(i)
                    col.append(word_index)
                    data.append(np.int8(1))
                ids.append(id)
                i += 1
    X = coo_matrix(
        (data, (row, col)),
        shape=(len(ids), get_n_features()),
        dtype=np.int8
    )
    return (X.tocsr(), ids)


def get_id_probs(id_features):
    """ Return prediction probabilities. """
    X, id_indexes = get_id_features_matrix(id_features)
    model = get_model()
    id_probs = {}
    y_prob = model.predict_proba(X)
    for i, row in enumerate(y_prob):
        id = id_indexes[i]
        id_probs[id] = row
    return id_probs


def get_id_probs_from_file(infile, encoding=ENCODING):
    """
    Calculate probabilities for each item found in infile.
    """
    id_features = get_id_features(infile, encoding=encoding)
    return get_id_probs(id_features.items())


def print_id_probs(id_probs):
    """
    Print out id, probs items.
    """
    for id, probs in id_probs:
        probs_str = DELIMITER.join([str(p) for p in probs])
        print("{0}{1}{2}".format(id, DELIMITER, probs_str))


def main(arguments):
    """
    Print out rlev probabilities for data from input file.
    """
    encoding = ENCODING
    if arguments['--encoding']:
        encoding = arguments['--encoding']
    id_probs = get_id_probs_from_file(arguments['<infile>'], encoding=encoding)
    print_id_probs(id_probs.items())


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
