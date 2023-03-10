from __future__ import absolute_import, division, print_function

import os
import pickle
import gzip
import argparse
import statistics
import statistics
from collections import OrderedDict

from utils import *
from data_utils import AmazonDataset
#from knowledge_graph import KnowledgeGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {BEAUTY, CELL, CD, CLOTH}.')
    args = parser.parse_args()

    # Create AmazonDataset instance for dataset.
    # ========== BEGIN ========== #
    print('Load', args.dataset, 'train dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = AmazonDataset(DATASET_DIR[args.dataset], 'train')
    save_dataset(args.dataset, dataset, 'train')

    # Create AmazonDataset instance for dataset.
    # ========== BEGIN ========== #
    print('Load', args.dataset, 'test dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
    dataset = AmazonDataset(DATASET_DIR[args.dataset], 'test')
    save_dataset(args.dataset, dataset, 'test')

    # Generate knowledge graph instance.
    # ========== BEGIN ========== #
    print('Create', args.dataset, 'knowledge graph from dataset...')
    dataset = load_dataset(args.dataset)
    dataset_test = load_dataset(args.dataset, 'test')

    print('User: size: ', len(dataset.user.vocab))
    print('Product: size: ', len(dataset.product.vocab))
    print('word: size: ', len(dataset.word.vocab))
    print('related_product: size: ', len(dataset.related_product.vocab))
    print('brand: size: ', len(dataset.brand.vocab))
    print('category: size: ', len(dataset.category.vocab))

    #### User Purchases Products
    user = [item[0] for item in dataset.review.data]
    user_test = [item[0] for item in dataset_test.review.data]
    user.extend(user_test)
    #print('review: user: total', len(user))
    freq = {}
    for userId in user:
        freq[userId] = freq.get(userId, 0) + 1
    freq1 = OrderedDict(sorted(freq.items()))
    user_freq = list(freq1.values())
    print('review: user: stats: mean: ', statistics.mean(user_freq))
    print('review: user: stats: std: ', statistics.pstdev(user_freq))

    #### User mention feature word
    userMentionWord = [(item[0], len(item[2])) for item in dataset.review.data]
    userMentionWord_test = [(item[0], len(item[2])) for item in dataset_test.review.data]

    userMentionWord.extend(userMentionWord_test)
    tup = {i: 0 for i, v in userMentionWord}
    for key, value in userMentionWord:
        tup[key] = tup[key] + value

    result = list(map(tuple, tup.items()))
    result = OrderedDict(sorted(result))
    #print('review: user: mention feature word total', len(result))

    List_userMentionWord = list(result.values())
    print('review: user: mention feature word: stats: mean: ', statistics.mean(List_userMentionWord))
    print('review: user: mention feature word: stats: std: ', statistics.pstdev(List_userMentionWord))

    #### Product mention feature word
    prodMentionWord = [(item[1], len(item[2])) for item in dataset.review.data]
    prodMentionWord_test = [(item[1], len(item[2])) for item in dataset_test.review.data]

    prodMentionWord.extend(prodMentionWord_test)
    tup = {i: 0 for i, v in prodMentionWord}
    for key, value in prodMentionWord:
        tup[key] = tup[key] + value

    result = list(map(tuple, tup.items()))
    result = OrderedDict(sorted(result))
    #print('review: product: mention feature word total', len(result))

    List_prodMentionWord = list(result.values())
    print('review: product: mention feature word: stats: mean: ', statistics.mean(List_prodMentionWord))
    print('review: product: mention feature word: stats: std: ', statistics.pstdev(List_prodMentionWord))


    d = [len(list) for list in dataset.belongs_to.data]
    #print('belongs_to: stats: mean: ', len(d))
    print('belongs_to: stats: mean: ', round(statistics.mean(d), 2))
    print('belongs_to: stats: std: ', round(statistics.pstdev(d), 2))

    d = [len(list) for list in dataset.produced_by.data]
    #print('produced_by: stats: mean: ', len(d))
    print('produced_by: stats: mean: ', round(statistics.mean(d), 2))
    print('produced_by: stats: std: ', round(statistics.pstdev(d), 2))

    d = [len(list) for list in dataset.also_bought.data]
    #print('also_bought: stats: mean: ', len(d))
    print('also_bought: stats: mean: ', round(statistics.mean(d), 2))
    print('also_bought: stats: std: ', round(statistics.pstdev(d), 2))

    d = [len(list) for list in dataset.also_viewed.data]
    #print('also_viewed: stats: mean: ', len(d))
    print('also_viewed: stats: mean: ', round(statistics.mean(d), 2))
    print('also_viewed: stats: std: ', round(statistics.pstdev(d), 2))

    d = [len(list) for list in dataset.bought_together.data]
    #print('bought_together: stats: mean: ', len(d))
    print('bought_together: stats: mean: ', round(statistics.mean(d), 2))
    print('bought_together: stats: std: ', round(statistics.pstdev(d), 2))



# =========== END =========== #


if __name__ == '__main__':
    main()

