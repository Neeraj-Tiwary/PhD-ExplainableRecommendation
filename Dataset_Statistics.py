from __future__ import absolute_import, division, print_function

import os
import pickle
import gzip
import argparse
import statistics
from tabulate import tabulate
from collections import OrderedDict

from utils import *
from data_utils import AmazonDataset


def get_relation_stats(source_targets):
    source_freq, target_freq = {}, {}
    for source_target in source_targets:
        source_freq[source_target[0]] = source_freq.get(source_target[0], 0) + 1
        if type(source_target[1]) == int:
            target_freq[source_target[1]] = target_freq.get(source_target[1], 0) + 1
        else:
            for item in source_target[1]:
                target_freq[item] = target_freq.get(item, 0) + 1
    freq1 = OrderedDict(sorted(source_freq.items()))
    freq2 = OrderedDict(sorted(target_freq.items()))
    source_freq = list(freq1.values())
    target_freq = list(freq2.values())
    return source_freq, target_freq


def get_review_stats(source_targets):
    tup = {i: 0 for i, v in source_targets}
    for key, value in source_targets:
        tup[key] = tup[key] + value

    result = list(map(tuple, tup.items()))
    result = OrderedDict(sorted(result))
    return list(result.values())


def stats(args):
    # Load AmazonDataset instance for dataset.
    # ========== Train ========== #
    print('Load', args.dataset, 'train dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        os.makedirs(TMP_DIR[args.dataset])
        dataset = AmazonDataset(DATASET_DIR[args.dataset], 'train')
        save_dataset(args.dataset, dataset, 'train')

    # =========== Test ========== #
    print('Load', args.dataset, 'test dataset from file...')
    if not os.path.isdir(TMP_DIR[args.dataset]):
        dataset = AmazonDataset(DATASET_DIR[args.dataset], 'test')
        save_dataset(args.dataset, dataset, 'test')

    # Generate stats
    # ========== BEGIN ========== #
    print('Generate Stats : ', args.dataset)
    dataset = load_dataset(args.dataset)
    dataset_test = load_dataset(args.dataset, 'test')

    # Define the headers and rows of the table
    headers = ["Entity", "Size"]
    rows = [
        ["User", len(dataset.user.vocab)],
        ["Product", len(dataset.product.vocab)],
        ["word", len(dataset.word.vocab)],
        ["related_product", len(dataset.related_product.vocab)],
        ["brand", len(dataset.brand.vocab)],
        ["category", len(dataset.category.vocab)]
    ]

    # Generate the table using tabulate module
    table = tabulate(rows, headers=headers)

    # Print the table
    print(table)

    # Define the headers and rows of the table
    headers = ["Relation", "Size", "Source Size", "Target Size", "Source Freq Mean", "Source Freq Std", "Target Freq Mean", "Target Freq Std", "%"]
    rows = []

    # =========== User Purchases Products ========== #
    user_products_train = [(item[0], item[1]) for item in dataset.review.data]
    user_products_test = [(item[0], item[1]) for item in dataset_test.review.data]
    user_products = user_products_train.copy()
    user_products.extend(user_products_test)

    # Get user prod stats
    user_freq, prod_freq = get_relation_stats(user_products)
    user_freq_train, prod_freq_train = get_relation_stats(user_products_train)
    user_freq_test, prod_freq_test = get_relation_stats(user_products_test)

    rows.append(["user_products: ", len(user_products), len(user_freq), len(prod_freq), round(statistics.mean(user_freq), 2), round(statistics.pstdev(user_freq), 2), round(statistics.mean(prod_freq), 2), round(statistics.pstdev(prod_freq), 2), ((len(user_products) / len(user_products)) * 100.0)])
    rows.append(["user_products_train: ", len(user_products_train), len(user_freq_train), len(prod_freq_train), round(statistics.mean(user_freq_train), 2), round(statistics.pstdev(user_freq_train), 2), round(statistics.mean(prod_freq_train), 2), round(statistics.pstdev(prod_freq_train), 2), round(((len(user_products_train) / len(user_products)) * 100.0), 2)])
    rows.append(["user_products_test: ", len(user_products_test), len(user_freq_test), len(prod_freq_test), round(statistics.mean(user_freq_test), 2), round(statistics.pstdev(user_freq_test), 2), round(statistics.mean(prod_freq_test), 2), round(statistics.pstdev(prod_freq_test), 2), round(((len(user_products_test) / len(user_products)) * 100.0), 2)])

    # =========== User mention feature word ========== #
    user_mention_word_train = [(item[0], item[2]) for item in dataset.review.data]
    user_mention_word_test = [(item[0], item[2]) for item in dataset_test.review.data]
    user_mention_word = user_mention_word_train.copy()
    user_mention_word.extend(user_mention_word_test)

    # Get user word stats
    user_freq, word_freq = get_relation_stats(user_mention_word)
    user_freq_train, word_freq_train = get_relation_stats(user_mention_word_train)
    user_freq_test, word_freq_test = get_relation_stats(user_mention_word_test)

    rows.append(["user_mention_word: ", len(user_mention_word), len(user_freq), len(word_freq), round(statistics.mean(user_freq), 2), round(statistics.pstdev(user_freq), 2), round(statistics.mean(word_freq), 2), round(statistics.pstdev(word_freq), 2), ((len(user_mention_word) / len(user_mention_word)) * 100.0)])
    rows.append(["user_mention_word_train: ", len(user_mention_word_train), len(user_freq_train), len(word_freq_train), round(statistics.mean(user_freq_train), 2), round(statistics.pstdev(user_freq_train), 2), round(statistics.mean(word_freq_train), 2), round(statistics.pstdev(word_freq_train), 2), round(((len(user_mention_word_train) / len(user_mention_word)) * 100.0), 2)])
    rows.append(["user_mention_word_test: ", len(user_mention_word_test), len(user_freq_test), len(word_freq_test), round(statistics.mean(user_freq_test), 2), round(statistics.pstdev(user_freq_test), 2), round(statistics.mean(word_freq_test), 2), round(statistics.pstdev(word_freq_test), 2), round(((len(user_mention_word_test) / len(user_mention_word)) * 100.0), 2)])

    # =========== User mention feature word - Len ========== #
    user_mention_word_train = [(item[0], len(item[2])) for item in dataset.review.data]
    user_mention_word_test = [(item[0], len(item[2])) for item in dataset_test.review.data]
    user_mention_word = user_mention_word_train.copy()
    user_mention_word.extend(user_mention_word_test)

    # Get user word stats
    word_freq = get_review_stats(user_mention_word)
    word_freq_train = get_review_stats(user_mention_word_train)
    word_freq_test = get_review_stats(user_mention_word_test)

    rows.append(["user_mention_word - len: ", len(user_mention_word), len(user_freq), len(word_freq), round(statistics.mean(user_freq), 2), round(statistics.pstdev(user_freq), 2), round(statistics.mean(word_freq), 2), round(statistics.pstdev(word_freq), 2), ((len(user_mention_word) / len(user_mention_word)) * 100.0)])
    rows.append(["user_mention_word_train - len: ", len(user_mention_word_train), len(user_freq_train), len(word_freq_train), round(statistics.mean(user_freq_train), 2), round(statistics.pstdev(user_freq_train), 2), round(statistics.mean(word_freq_train), 2), round(statistics.pstdev(word_freq_train), 2), round(((len(user_mention_word_train) / len(user_mention_word)) * 100.0), 2)])
    rows.append(["user_mention_word_test - len: ", len(user_mention_word_test), len(user_freq_test), len(word_freq_test), round(statistics.mean(user_freq_test), 2), round(statistics.pstdev(user_freq_test), 2), round(statistics.mean(word_freq_test), 2), round(statistics.pstdev(word_freq_test), 2), round(((len(user_mention_word_test) / len(user_mention_word)) * 100.0), 2)])

    # =========== Product mention feature word ========== #
    prod_described_word_train = [(item[1], item[2]) for item in dataset.review.data]
    prod_described_word_test = [(item[1], item[2]) for item in dataset_test.review.data]
    prod_described_word = prod_described_word_train.copy()
    prod_described_word.extend(prod_described_word_test)

    # Get prod word stats
    prod_freq, word_freq = get_relation_stats(prod_described_word)
    prod_freq_train, word_freq_train = get_relation_stats(prod_described_word_train)
    prod_freq_test, word_freq_test = get_relation_stats(prod_described_word_test)

    rows.append(["prod_described_word: ", len(prod_described_word), len(prod_freq), len(word_freq), round(statistics.mean(prod_freq), 2), round(statistics.pstdev(prod_freq), 2), round(statistics.mean(word_freq), 2), round(statistics.pstdev(word_freq), 2), ((len(prod_described_word) / len(prod_described_word)) * 100.0)])
    rows.append(["prod_described_word_train: ", len(prod_described_word_train), len(prod_freq_train), len(word_freq_train), round(statistics.mean(prod_freq_train), 2), round(statistics.pstdev(prod_freq_train), 2), round(statistics.mean(word_freq_train), 2), round(statistics.pstdev(word_freq_train), 2), round(((len(prod_described_word_train) / len(prod_described_word)) * 100.0), 2)])
    rows.append(["prod_described_word_test: ", len(prod_described_word_test), len(prod_freq_test), len(word_freq_test), round(statistics.mean(prod_freq_test), 2), round(statistics.pstdev(prod_freq_test), 2), round(statistics.mean(word_freq_test), 2), round(statistics.pstdev(word_freq_test), 2), round(((len(prod_described_word_test) / len(prod_described_word)) * 100.0), 2)])

    # =========== Product mention feature word len ========== #
    prod_described_word_train = [(item[1], len(item[2])) for item in dataset.review.data]
    prod_described_word_test = [(item[1], len(item[2])) for item in dataset_test.review.data]
    prod_described_word = prod_described_word_train.copy()
    prod_described_word.extend(prod_described_word_test)

    # Get prod word stats
    word_freq = get_review_stats(prod_described_word)
    word_freq_train = get_review_stats(prod_described_word_train)
    word_freq_test = get_review_stats(prod_described_word_test)

    rows.append(["prod_described_word - len: ", len(prod_described_word), len(prod_freq), len(word_freq), round(statistics.mean(prod_freq), 2), round(statistics.pstdev(prod_freq), 2), round(statistics.mean(word_freq), 2), round(statistics.pstdev(word_freq), 2), ((len(prod_described_word) / len(prod_described_word)) * 100.0)])
    rows.append(["prod_described_word_train - len: ", len(prod_described_word_train), len(prod_freq_train), len(word_freq_train), round(statistics.mean(prod_freq_train), 2), round(statistics.pstdev(prod_freq_train), 2), round(statistics.mean(word_freq_train), 2), round(statistics.pstdev(word_freq_train), 2), round(((len(prod_described_word_train) / len(prod_described_word)) * 100.0), 2)])
    rows.append(["prod_described_word_test - len: ", len(prod_described_word_test), len(prod_freq_test), len(word_freq_test), round(statistics.mean(prod_freq_test), 2), round(statistics.pstdev(prod_freq_test), 2), round(statistics.mean(word_freq_test), 2), round(statistics.pstdev(word_freq_test), 2), round(((len(prod_described_word_test) / len(prod_described_word)) * 100.0), 2)])

    # Generate the table using tabulate module
    table_stats = tabulate(rows, headers=headers)
    print(table_stats)

    # =========== Other Relations ========== #
    # Define the headers and rows of the table
    headers = ["Relation", "Size", "Mean", "Std"]
    rows = []
    d = [len(list) for list in dataset.belongs_to.data]
    rows.append(["belongs_to", len(d), round(statistics.mean(d), 2), round(statistics.pstdev(d), 2)])

    d = [len(list) for list in dataset.produced_by.data]
    rows.append(["produced_by", len(d), round(statistics.mean(d), 2), round(statistics.pstdev(d), 2)])

    d = [len(list) for list in dataset.also_bought.data]
    rows.append(["also_bought", len(d), round(statistics.mean(d), 2), round(statistics.pstdev(d), 2)])

    d = [len(list) for list in dataset.also_viewed.data]
    rows.append(["also_viewed", len(d), round(statistics.mean(d), 2), round(statistics.pstdev(d), 2)])

    d = [len(list) for list in dataset.bought_together.data]
    rows.append(["bought_together", len(d), round(statistics.mean(d), 2), round(statistics.pstdev(d), 2)])

    # Generate the table using tabulate module
    table_stats = tabulate(rows, headers)
    print(table_stats)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=CD, help='One of {BEAUTY, CELL, CD, CLOTH}.')
    parser.add_argument('--target_folder', type=str, default="Stats", help='One of {BEAUTY, CELL, CD, CLOTH}.')

    args = parser.parse_args()
    # Call stats function to generate the stats of the dataset
    stats(args)


if __name__ == '__main__':
    main()

