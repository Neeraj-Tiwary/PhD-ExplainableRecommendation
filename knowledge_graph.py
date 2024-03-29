from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch

from utils import *
from data_utils import AmazonDataset


class KnowledgeGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_reviews(dataset)
        self._load_knowledge(dataset)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities():
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: [] for r in get_relations(entity)}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_reviews(self, dataset, word_tfidf_threshold=0.2, word_freq_threshold=5000):
        print('Load reviews...')
        # (1) Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        reviews = [d[2] for d in dataset.review.data]
        review_tfidf = compute_tfidf_fast(vocab, reviews)
        distrib = dataset.review.word_distrib
        num_edges = 0
        all_removed_words = []
        for rid, data in enumerate(dataset.review.data):
            uid, pid, review = data
            doc_tfidf = review_tfidf[rid].toarray()[0]
            remained_words = [wid for wid in set(review)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(review).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            if len(remained_words) <= 0:
                continue

            # (2) Add edges.
            self._add_edge(USER, uid, PURCHASE, PRODUCT, pid)
            num_edges += 2
            for wid in remained_words:
                self._add_edge(USER, uid, MENTION, WORD, wid)
                self._add_edge(PRODUCT, pid, DESCRIBED_AS, WORD, wid)
                num_edges += 4
        print('Total {:d} review edges.'.format(num_edges))

       # with open('./tmp/review_removed_words.txt', 'w') as f:
       #     f.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def _load_knowledge(self, dataset):
        for relation in [PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for pid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(PRODUCT, relation)
                    self._add_edge(PRODUCT, pid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """ Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))
        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def set_top_matches(self, u_u_match, u_p_match, u_w_match):
        self.top_matches = {
            USER: u_u_match,
            PRODUCT: u_p_match,
            WORD: u_w_match,
        }

    def heuristic_search(self, uid, pid, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        if trim_edges:
            _get = lambda e, i, r: self.get_tails_given_user(e, i, r, uid)
        else:
            _get = lambda e, i, r: self.get_tails(e, i, r)
        pattern = PATH_PATTERN[pattern_id]
        paths = []
        if pattern_id == 1:  # OK
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            wids_p = set(_get(PRODUCT, pid, DESCRIBED_AS))  # PRODUCT->DESCRIBE->WORD
            intersect_nodes = wids_u.intersection(wids_p)
            paths = [(uid, x, pid) for x in intersect_nodes]
        elif pattern_id in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            pids_u = set(_get(USER, uid, PURCHASE))  # USER->PURCHASE->PRODUCT
            pids_u = pids_u.difference([pid])  # exclude target product
            nodes_p = set(_get(PRODUCT, pid, pattern[3][0]))  # PRODUCT->relation->node2
            if pattern[2][1] == USER:
                nodes_p = nodes_p.difference(set([uid]))  # remove the self user
            elif pattern[2][1] == CATEGORY:
                nodes_p = nodes_p.difference(set([0]))  # remove the most common category
            for pid_u in pids_u:
                relation, entity_tail = pattern[2][0], pattern[2][1]
                et_ids = set(_get(PRODUCT, pid_u, relation))  # USER->PURCHASE->PRODUCT->relation->node2
                intersect_nodes = et_ids.intersection(nodes_p)
                tmp_paths = [(uid, pid_u, x, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        elif pattern_id == 10:
            wids_u = set(_get(USER, uid, MENTION))  # USER->MENTION->WORD
            uids_p = set(_get(PRODUCT, pid, PURCHASE))  # PRODUCT->PURCHASE->USER
            uids_p = uids_p.difference([uid])  # exclude source user
            for uid_p in uids_p:
                wids_u_p = set(_get(USER, uid_p, MENTION))  # PRODUCT->PURCHASE->USER->MENTION->WORD
                intersect_nodes = wids_u.intersection(wids_u_p)
                tmp_paths = [(uid, x, uid_p, pid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        return paths


def check_user_product_path(dataset_str, kg, batch_uids=None, mode='train', is_debug=0):
    # Check if there exists at least one path for any user-product in test set.
    user_products_labels = load_labels(dataset_str, mode)
    if len(batch_uids) != 0:
        user_products_labels = dict(sorted((key, batch_uids[key]) for key in batch_uids.keys() if key in batch_uids))
    if is_debug == 1:
        print('user_products_labels: ', user_products_labels)
    user_labels_path = []
    for uid in user_products_labels:
        for pid in user_products_labels[uid]:
            #print(uid, pid)
            count = 0
            for pattern_id in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
                tmp_path = kg.heuristic_search(uid, pid, pattern_id)
                user_labels_path.append({'uid': uid, 'pid': pid, 'pattern_id': pattern_id, 'count_paths': len(tmp_path), 'path': tmp_path})
                count += len(tmp_path)

    print(user_labels_path)
    return user_labels_path


def batch_get_user_product_path_actions_actual(dataset_str, kg, batch_uids_targetlabel={}, batch_path=[], batch_curr_further_processing=[], is_train=0, is_debug=0):
    # Check if there exists at least one path for any user-product in the dataset.
    if is_train == 1:
        mode = 'train'
    else:
        mode = 'test'
    if len(batch_uids_targetlabel) != 0:
        user_products_labels = dict((key, [batch_uids_targetlabel[key]]) for key in batch_uids_targetlabel.keys() if key in batch_uids_targetlabel)
    else:
        user_products_labels = load_labels(dataset_str, mode)
    if is_debug == 1:
        print('user_products_labels: ', user_products_labels)
        print('batch_curr_further_processing: ', batch_curr_further_processing)
    batch_user_acts_actual = {}
    for uid in user_products_labels:
        user_acts = []  # list of tuples of (relation, node_type, node_id)
        if is_debug == 1:
            print('uid: ', uid)
            print('batch_curr_further_processing[uid]: ', batch_curr_further_processing[uid])
        if batch_curr_further_processing[uid] == 1:
            for pid in user_products_labels[uid]:
                if pid != -9999:
                    path = [path for path in batch_path if path[0][2] == uid][0]
                    # path = batch_path[uid]
                    if is_debug == 1:
                        print('path: ', path)
                    for pattern_id in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
                        if is_debug == 1:
                            print(PATH_PATTERN[pattern_id])
                        pattern = tuple([SELF_LOOP] + [v[0] for v in PATH_PATTERN[pattern_id][1:]])
                        path_pattern = tuple([v[0] for v in path])
                        if is_debug == 1:
                            print('path_pattern:', path_pattern)
                            print('pattern:', pattern)
                            print('uid, pid, path, pattern_id: ', uid, pid, path, pattern_id)
                        if path_pattern <= pattern:
                            tmp_path = kg.heuristic_search(uid=uid, pid=pid, pattern_id=pattern_id)
                            if is_debug == 1:
                                print('inside')
                                print('tmp_path:', tmp_path)
                            if len(tmp_path) > 0:
                                _, curr_node_type, curr_node_id = path[-1]
                                relations_nodes = kg(curr_node_type, curr_node_id)
                                for r in relations_nodes:
                                    path_pattern_ext = path_pattern + (r,)
                                    if is_debug == 1:
                                        print('r:', path_pattern_ext)
                                    if any((path_pattern_ext == pattern[i:i + len(path_pattern_ext)]) for i in range(len(pattern) - len(path_pattern_ext) + 1)):
                                        next_node_type = KG_RELATION[curr_node_type][r]
                                        next_node_ids = relations_nodes[r]
                                        if is_debug == 1:
                                            print('r:', r, 'next_node_type:', next_node_type, 'next_node_ids: Pre:', next_node_ids)
                                        next_node_ids = [n for n in next_node_ids if n in [v[len(path)] for v in tmp_path]]  # filter
                                        if is_debug == 1:
                                            print('r:', r, 'next_node_type:', next_node_type, 'next_node_ids: Post:', next_node_ids)
                                        user_acts.extend(zip([r] * len(next_node_ids), next_node_ids))
        batch_user_acts_actual[uid] = sorted(set(user_acts))
    if is_debug == 1:
        print(batch_user_acts_actual)
    return batch_user_acts_actual


def batch_get_user_products_actions_actual(dataset_str, kg, batch_uids_targetlabel={}, batch_path=[], batch_curr_further_processing=[], is_train=0, is_debug=0):
    # Check if there exists at least one path for any user-product in the dataset.
    if is_train == 1:
        mode = 'train'
    else:
        mode = 'test'
    if len(batch_uids_targetlabel) != 0:
        user_products_labels = dict((key, batch_uids_targetlabel[key]) for key in batch_uids_targetlabel.keys() if key in batch_uids_targetlabel)
    else:
        user_products_labels = load_labels(dataset_str, mode)
    if is_debug == 1:
        print('user_products_labels: ', user_products_labels)
        print('batch_curr_further_processing: ', batch_curr_further_processing)
    batch_user_acts_actual = []
    for uid in user_products_labels:
        user_pid_acts = {}
        user_acts_list = []
        if is_debug == 1:
            print('uid: ', uid)
            print('batch_curr_further_processing[uid]: ', batch_curr_further_processing[uid])
        if batch_curr_further_processing[uid] == 1:
            for pid in user_products_labels[uid]:
                user_acts = []  # list of tuples of (relation, node_type, node_id)
                if pid != -9999:
                    path = [path for path in batch_path if path[0][2] == uid][0]
                    # path = batch_path[uid]
                    if is_debug == 1:
                        print('path: ', path)
                    for pattern_id in [1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
                        if is_debug == 1:
                            print(PATH_PATTERN[pattern_id])
                        pattern = tuple([SELF_LOOP] + [v[0] for v in PATH_PATTERN[pattern_id][1:]])
                        path_pattern = tuple([v[0] for v in path])
                        if is_debug == 1:
                            print('path_pattern:', path_pattern)
                            print('pattern:', pattern)
                            print('uid, pid, path, pattern_id: ', uid, pid, path, pattern_id)
                        if path_pattern <= pattern:
                            tmp_path = kg.heuristic_search(uid=uid, pid=pid, pattern_id=pattern_id)
                            if is_debug == 1:
                                print('inside')
                                print('tmp_path:', tmp_path)
                            if len(tmp_path) > 0:
                                _, curr_node_type, curr_node_id = path[-1]
                                relations_nodes = kg(curr_node_type, curr_node_id)
                                for r in relations_nodes:
                                    path_pattern_ext = path_pattern + (r,)
                                    if is_debug == 1:
                                        print('r:', path_pattern_ext)
                                    if any((path_pattern_ext == pattern[i:i + len(path_pattern_ext)]) for i in range(len(pattern) - len(path_pattern_ext) + 1)):
                                        next_node_type = KG_RELATION[curr_node_type][r]
                                        next_node_ids = relations_nodes[r]
                                        if is_debug == 1:
                                            print('r:', r, 'next_node_type:', next_node_type, 'next_node_ids: Pre:', next_node_ids)
                                        next_node_ids = [n for n in next_node_ids if n in [v[len(path)] for v in tmp_path]]  # filter
                                        if is_debug == 1:
                                            print('r:', r, 'next_node_type:', next_node_type, 'next_node_ids: Post:', next_node_ids)
                                        user_acts.extend(zip([r] * len(next_node_ids), next_node_ids))
                user_pid_acts[pid] = sorted(set(user_acts))
        user_acts_list = sorted(list(set([acts for pid_acts in user_pid_acts.values() for acts in pid_acts])))
        user_acts_list = [('self_loop', uid)] + user_acts_list
        batch_user_acts_actual.append(user_acts_list) #user_pid_acts
    if is_debug == 1:
        print(batch_user_acts_actual)
    return batch_user_acts_actual
