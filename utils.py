from __future__ import absolute_import, division, print_function

import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch
import glob
import os.path

# Dataset names.
BEAUTY = 'beauty'
CELL = 'cell'
CLOTH = 'cloth'
CD = 'cd'

# Dataset directories.
DATASET_DIR = {
    BEAUTY: './data/Amazon_Beauty',
    CELL: './data/Amazon_Cellphones',
    CLOTH: './data/Amazon_Clothing',
    CD: './data/Amazon_CDs',
}

# Model result directories.
TMP_DIR = {
    BEAUTY: './tmp/Amazon_Beauty',
    CELL: './tmp/Amazon_Cellphones',
    CLOTH: './tmp/Amazon_Clothing',
    CD: './tmp/Amazon_CDs',
}

# Label files.
LABELS = {
    BEAUTY: (TMP_DIR[BEAUTY] + '/train_label.pkl', TMP_DIR[BEAUTY] + '/test_label.pkl'),
    CLOTH: (TMP_DIR[CLOTH] + '/train_label.pkl', TMP_DIR[CLOTH] + '/test_label.pkl'),
    CELL: (TMP_DIR[CELL] + '/train_label.pkl', TMP_DIR[CELL] + '/test_label.pkl'),
    CD: (TMP_DIR[CD] + '/train_label.pkl', TMP_DIR[CD] + '/test_label.pkl')
}

# Entities
USER = 'user'
PRODUCT = 'product'
WORD = 'word'
RPRODUCT = 'related_product'
BRAND = 'brand'
CATEGORY = 'category'

# Relations
PURCHASE = 'purchase'
MENTION = 'mentions'
DESCRIBED_AS = 'described_as'
PRODUCED_BY = 'produced_by'
BELONG_TO = 'belongs_to'
ALSO_BOUGHT = 'also_bought'
ALSO_VIEWED = 'also_viewed'
BOUGHT_TOGETHER = 'bought_together'
SELF_LOOP = 'self_loop'  # only for kg env

# Defining the KG relation
KG_RELATION = {
    USER: {
        PURCHASE: PRODUCT,
        MENTION: WORD,
    },
    WORD: {
        MENTION: USER,
        DESCRIBED_AS: PRODUCT,
    },
    PRODUCT: {
        PURCHASE: USER,
        DESCRIBED_AS: WORD,
        PRODUCED_BY: BRAND,
        BELONG_TO: CATEGORY,
        ALSO_BOUGHT: RPRODUCT,
        ALSO_VIEWED: RPRODUCT,
        BOUGHT_TOGETHER: RPRODUCT,
    },
    BRAND: {
        PRODUCED_BY: PRODUCT,
    },
    CATEGORY: {
        BELONG_TO: PRODUCT,
    },
    RPRODUCT: {
        ALSO_BOUGHT: PRODUCT,
        ALSO_VIEWED: PRODUCT,
        BOUGHT_TOGETHER: PRODUCT,
    }
}

# Defining the followed path patterns in the KG
PATH_PATTERN = {
    # length = 3 - Final path
    1:  ((None, USER), (MENTION, WORD), (DESCRIBED_AS, PRODUCT)),
    # length = 3 - sub paths
    2:  ((None, USER), (MENTION, WORD), (MENTION, USER)),
    3:  ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER)),
    4:  ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD)),
    5:  ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND)),
    6:  ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY)),
    7:  ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT)),
    8:  ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT)),
    9:  ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT)),
    # length = 4 - Final paths
    10: ((None, USER), (MENTION, WORD), (MENTION, USER), (PURCHASE, PRODUCT)),
    11: ((None, USER), (PURCHASE, PRODUCT), (PURCHASE, USER), (PURCHASE, PRODUCT)),
    12: ((None, USER), (PURCHASE, PRODUCT), (DESCRIBED_AS, WORD), (DESCRIBED_AS, PRODUCT)),
    13: ((None, USER), (PURCHASE, PRODUCT), (PRODUCED_BY, BRAND), (PRODUCED_BY, PRODUCT)),
    14: ((None, USER), (PURCHASE, PRODUCT), (BELONG_TO, CATEGORY), (BELONG_TO, PRODUCT)),
    15: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    16: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    17: ((None, USER), (PURCHASE, PRODUCT), (ALSO_BOUGHT, RPRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
    18: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    19: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    20: ((None, USER), (PURCHASE, PRODUCT), (ALSO_VIEWED, RPRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
    21: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT), (ALSO_BOUGHT, PRODUCT)),
    22: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT), (ALSO_VIEWED, PRODUCT)),
    23: ((None, USER), (PURCHASE, PRODUCT), (BOUGHT_TOGETHER, RPRODUCT), (BOUGHT_TOGETHER, PRODUCT)),
}

''' ######################### Defining the utilities functions ######################### '''


def get_entities():
    return list(KG_RELATION.keys())


def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname, mode='w'):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj, mode='train'):
    if mode == 'train':
        dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    else:
        dataset_file = TMP_DIR[dataset] + '/dataset' + '_' + mode + '.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset, mode='train'):
    if mode == 'train':
        dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    else:
        dataset_file = TMP_DIR[dataset] + '/dataset' + '_' + mode + '.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg


def get_entity_details(dataset, entity_type, entity_id):
    if entity_type == 'user':
        entity_value = dataset.user.vocab[entity_id]
    elif entity_type == 'product':
        entity_value = dataset.product.vocab[entity_id]
    elif entity_type == 'word':
        entity_value = dataset.word.vocab[entity_id]
    elif entity_type == 'related_product':
        entity_value = dataset.related_product.vocab[entity_id]
    elif entity_type == 'brand':
        entity_value = dataset.brand.vocab[entity_id]
    elif entity_type == 'category':
        entity_value = dataset.category.vocab[entity_id]
    else:
        entity_value = None
    return entity_value


def load_checkpoint(checkpoint_fpath):
    checkpoint = torch.load(checkpoint_fpath, map_location=lambda storage, loc: storage)
    return checkpoint


def save_checkpoint(checkpoint, checkpoint_fpath):
    torch.save(checkpoint, checkpoint_fpath)


def get_latest_file(folder_path, file_type):
    files = glob.glob(folder_path + file_type)
    if len(files) > 0:
        max_file = max(files, key=os.path.getctime)
    else:
        max_file = None
    return max_file
