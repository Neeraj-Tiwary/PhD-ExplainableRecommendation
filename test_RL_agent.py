from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime

import numpy as np
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading
from functools import reduce
from operator import mul

import knowledge_graph
from knowledge_graph import *
from kg_env import BatchKGEnvironment
from train_RL_agent import ActorCritic
from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#global logger
#logger = None

# Function to get the quantification of the explainability
def get_explainability_score(pred_labels_details, args):
    """Compute explainability metrics for predicted recommendations.
    Args:
        pred_labels_details: A prediction label/s consisting of path score, probability, entropy, rewards and path of that prediction.

        Quantitative evaluation of explainability
        R - the number of rules outputted by the explanation
        S - The average score of the path traversal for the recommended item
        P - The average probability of the path traversal for the recommended item
        Rw- The average reward of the path traversal for the recommended item

        Formula = (S + P + Rw)/ ((MAX range(S) + MAX range(P) + MAX range(Rw))) * R
    """
    # Define variable
    explainability_score = 0

    # Extract the key metrics from the prediction label details
    pred_probs = pred_labels_details[1]                     # Probability of the prediction label
    pred_entropy = pred_labels_details[2]                   # Entropy of the prediction label
    pred_reward = pred_labels_details[3]                    # Rewards of the prediction label
    pred_path = pred_labels_details[4]                      # Path traverses by the user to reach to the prediction
    path_prob_diff_user_mean = pred_labels_details[6]       # Prob difference of the prediction from the user mean
    path_entropy_diff_user_mean = pred_labels_details[7]    # Entropy difference of the prediction from the user mean
    path_rewards_diff_user_mean = pred_labels_details[8]    # Rewards difference of the prediction from the user mean

    # Compute metrics for Explainability - Rewards * Entropy
    if args.MES_score_option == 1:
        explainability_score = (((pred_reward + path_rewards_diff_user_mean) / (pred_reward - path_rewards_diff_user_mean)) * (pred_entropy + path_entropy_diff_user_mean)) / len(pred_path)
    # Compute metrics for Explainability - Only Rewards
    elif args.MES_score_option == 2:
        explainability_score = ((pred_reward + path_rewards_diff_user_mean) / (pred_reward - path_rewards_diff_user_mean))
    # Compute metrics for Explainability - Only Entropy
    elif args.MES_score_option == 3:
        explainability_score = (pred_entropy + path_entropy_diff_user_mean)
    # Compute metrics for Explainability - Only Probs
    elif args.MES_score_option == 4:
        explainability_score = (pred_probs + path_prob_diff_user_mean)
    # Compute metrics for Explainability - Entropy *  Probs
    elif args.MES_score_option == 5:
        explainability_score = ((pred_entropy + path_entropy_diff_user_mean) * (pred_probs + path_prob_diff_user_mean))
    # Compute metrics for Explainability - Rewards *  Probs
    elif args.MES_score_option == 6:
        explainability_score = (((pred_reward + path_rewards_diff_user_mean) / (pred_reward - path_rewards_diff_user_mean)) * (pred_probs + path_prob_diff_user_mean))
    # Compute metrics for Explainability - Rewards *  Probs * Entropy
    elif args.MES_score_option == 7:
        explainability_score = (((pred_reward + path_rewards_diff_user_mean) / (pred_reward - path_rewards_diff_user_mean)) * (pred_probs + path_prob_diff_user_mean) * (pred_entropy + path_entropy_diff_user_mean)) / len(pred_path)
    # Compute metrics for Explainability - Rewards +  Probs + Entropy
    elif args.MES_score_option == 8:
        explainability_score = (((pred_reward + path_rewards_diff_user_mean) / (pred_reward - path_rewards_diff_user_mean)) + (pred_entropy + path_entropy_diff_user_mean) + (pred_probs + path_prob_diff_user_mean))
    #--- Baseline

    if args.debug == 1:
        print('pred_probs={} | pred_entropy={} | pred_reward={} | | pred_path={} | path_prob_diff_user_mean={} | path_entropy_diff_user_mean={} | path_rewards_diff_user_mean={} |  len(pred_path)={}'.
              format(pred_probs, pred_entropy, pred_reward, pred_path, path_prob_diff_user_mean, path_entropy_diff_user_mean, path_rewards_diff_user_mean, len(pred_path)))
        print('explainability_score: ', explainability_score)

    return explainability_score


# Function to get the quantification of the explainability
def get_product_prioritisation_score(pred_labels_details, args):
    """Compute explainability metrics for predicted recommendations.
    Args:
        pred_labels_details: A prediction label/s consisting of path score, probability, entropy, rewards and path of that prediction.

        Quantitative evaluation of explainability
        R - the number of rules outputted by the explanation
        S - The average score of the path traversal for the recommended item
        P - The average probability of the path traversal for the recommended item
        Rw- The average reward of the path traversal for the recommended item

        Formula = (S + P + Rw)/ ((MAX range(S) + MAX range(P) + MAX range(Rw))) * R
    """
    # Define variable
    #affinity_score = 0
    # Extract the key metrics from the prediction label details
    pred_score = pred_labels_details[0]                     # Affinity score of the prediction label
    pred_probs = pred_labels_details[1]                     # Probability of the prediction label
    pred_entropy = pred_labels_details[2]                   # Entropy of the prediction label
    pred_reward = pred_labels_details[3]                    # Rewards of the prediction label
    pred_path = pred_labels_details[4]                      # Path traverses by the user to reach to the prediction
    path_score_diff_user_mean = pred_labels_details[5]      # Affinity score difference of the prediction from the user mean
    path_prob_diff_user_mean = pred_labels_details[6]       # Prob difference of the prediction from the user mean
    path_entropy_diff_user_mean = pred_labels_details[7]    # Entropy difference of the prediction from the user mean
    path_rewards_diff_user_mean = pred_labels_details[8]    # Rewards difference of the prediction from the user mean

    # Compute metrics for Explainability - (Rewards Gain + Score Gain) * Entropy
    affinity_score = ((((pred_reward + path_rewards_diff_user_mean) / (pred_reward - path_rewards_diff_user_mean)) + (pred_score + path_score_diff_user_mean)) * (pred_entropy + path_entropy_diff_user_mean))

    if args.debug == 1:
        print('pred_score={} | pred_probs={} | pred_entropy={} | pred_reward={} | | pred_path={} | path_score_diff_user_mean={} | path_prob_diff_user_mean={} | path_entropy_diff_user_mean={} | path_rewards_diff_user_mean={} |  len(pred_path)={}'.
              format(pred_score, pred_probs, pred_entropy, pred_reward, pred_path, path_score_diff_user_mean, path_prob_diff_user_mean, path_entropy_diff_user_mean, path_rewards_diff_user_mean, len(pred_path)))
        print('affinity_score: ', affinity_score)

    return affinity_score


def evaluate(topk_matches, test_user_products, args):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    is_debug = args.debug
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::1], test_user_products[uid]
        if len(pred_list) == 0:
            continue
        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    if is_debug == 1:
        print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))
    return avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)


def batch_beam_search(env, model, uids, device, topk, args):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[1:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)
    is_debug = args.debug
    # initial reset state
    state_pool, path_pool, acts_pool, _, _, _ = env.reset(uids)  # numpy of [bs, dim]
    probs_pool = [[] for _ in uids]
    rewards_pool = [[] for _ in uids]
    # entropy_pool = [[] for _ in uids]
    '''if is_debug == 1:
        # print('state_pool: ', state_pool)
        print('path_pool:', path_pool)
        print('probs_pool:', probs_pool)
        print('rewards_pool:', rewards_pool)
        print('acts_pool:', acts_pool)'''

    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        #rewards = rewards_pool 
        '''if is_debug == 1:
            print('hop:', hop)
            print('state_pool: ', state_pool)
            print('state_tensor:', state_tensor)
            #print('acts_pool:', acts_pool)
            print('actmask_pool:', actmask_pool)
            print('actmask_tensor:', actmask_tensor)
            print('probs: Earlier: ', probs)'''
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        '''if is_debug == 1:
            print('probs: After: ', probs)
            print('topk[hop]:', topk[hop])'''
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()
        '''if is_debug == 1:
            print('topk_probs:', topk_probs, 'topk_idxs:', topk_idxs)
            print('topk_idxs.shape', topk_idxs.shape)'''

        new_path_pool, new_probs_pool, new_rewards_pool = [], [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            rewards = rewards_pool[row]
            if (len(rewards) >= 2 and sum(rewards) > 0) or (len(rewards) <= 1):
                '''if is_debug == 1:
                    print('row, path, probs: ', row, path, probs)
                    print('topk_idxs[row], topk_probs[row]:', topk_idxs[row], topk_probs[row])
                    print('rewards, sum, len: ', rewards, sum(rewards), len(rewards))'''
                for idx, p in zip(topk_idxs[row], topk_probs[row]):
                    if idx >= len(acts_pool[row]):  # act idx is invalid
                        continue
                    '''if is_debug == 1:
                        print('idx, p:', idx, p)
                        print('acts_pool[row]:', acts_pool[row])
                        print('acts_pool[row][idx]:', acts_pool[row][idx])'''
                    relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                    #print('relation:', relation)
                    if relation == SELF_LOOP:
                        next_node_type = path[-1][1]
                    else:
                        next_node_type = KG_RELATION[path[-1][1]][relation]
                    new_path = path + [(relation, next_node_type, next_node_id)]
                    '''if is_debug == 1:
                        print('next_node_type:', next_node_type)
                        print('new_path:', new_path)'''
                    new_rewards = env._get_reward(new_path, is_train=0, is_debug=0)
                    if new_rewards >= 0:
                        new_path_pool.append(new_path)
                        new_probs_pool.append(probs + [p])
                        new_rewards_pool.append(rewards + [new_rewards])

        path_pool = new_path_pool
        probs_pool = new_probs_pool
        rewards_pool = new_rewards_pool
        acts_pool = env._batch_get_actions(path_pool, False, 0, None, 0, 0)  # list of list, size=bs
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)
        if is_debug == 1:
            # print('state_pool: ', state_pool)
            #print('path_pool:', path_pool)
            #print('probs_pool:', probs_pool)
            #print('rewards_pool:', rewards_pool)
            print('shape: path_pool, probs_pool, rewards_pool : ', len(path_pool), len(probs_pool), len(rewards_pool))
            #print('acts_pool:', acts_pool)'''

    return path_pool, probs_pool, rewards_pool


def predict_paths(policy_file, path_file, test_labels, args):
    is_debug = args.debug
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = load_checkpoint(policy_file)['model_state_dict']

    if is_debug == 1:
        print('Start: predict_paths \nPredicting paths...')
        print('env.state_dim : ', env.state_dim)
        print('env.act_dim : ', env.act_dim)
        print('args.gamma : ', args.gamma)
        print('args.hidden : ', args.hidden)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_uids = list(test_labels.keys())
    batch_size = args.batch_size
    start_idx = 0
    all_paths, all_probs, all_entropy, all_rewards = [], [], [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs, rewards = batch_beam_search(env, model, uids=batch_uids, device=args.device, topk=args.topk, args=args)
        all_paths.extend(paths)
        all_probs.extend(probs)
        all_rewards.extend(rewards)

        # Entropy calculation
        probs_tensor = torch.FloatTensor(probs).to(args.device)
        entropy = Categorical(probs_tensor).entropy()
        entropy = entropy.detach().cpu().numpy()
        all_entropy.extend(entropy)

        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs, 'entropy': all_entropy, 'rewards': all_rewards}

    '''if is_debug == 1:
        print('paths : ', all_paths)
        print('max(paths) : ', max(all_paths), 'len(paths):', len(all_paths))
        print('probs : ', all_probs)
        print('max(probs) : ', max(all_probs), 'len(probs):', len(all_probs))
        print('entropy : ', all_entropy)
        print('max(entropy) : ', max(all_entropy), 'len(entropy):', len(all_entropy))
        print('rewards : ', all_rewards)
        print('max(rewards) : ', max(all_rewards), 'len(rewards):', len(all_rewards))
        print('predicts : ', predicts)'''
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(path_file, train_labels, test_labels, args, epoch, logger):
    is_debug = args.debug
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    purchase_embeds = embeds[PURCHASE][0]
    product_embeds = embeds[PRODUCT]
    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))

    pred_paths = {uid: {} for uid in test_labels}
    for path, probs, entropy, reward in zip(results['paths'], results['probs'], results['entropy'], results['rewards']):
        if path[-1][1] != PRODUCT:
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: (x * y), probs)
        path_entropy = entropy
        path_reward = sum(reward)
        pred_paths[uid][pid].append((path_score, path_prob, path_entropy, path_reward, path))

    # Find user wise mean of path_score, path_prob, path_entropy, path_reward
    user_path_mean_stats = {}
    for userid, products in pred_paths.items():
        user_path_mean_stats[userid] = []
        sum_path_score, sum_path_prob, sum_path_entropy, sum_path_reward, count = 0, 0, 0, 0, 0
        for pid, paths_info in pred_paths[userid].items():
            for path_info in paths_info:
                sum_path_score += path_info[0]
                sum_path_prob += path_info[1]
                sum_path_entropy += path_info[2]
                sum_path_reward += path_info[3]
                count += 1
        # to handle divide by zero error
        count = max(count, 1)
        user_path_mean_stats[userid].extend([float(sum_path_score/count), float(sum_path_prob/count), float(sum_path_entropy/count), float(sum_path_reward/count)])
        # print('user_path_mean_stats, count: ', user_path_mean_stats, count)

    # Find user wise difference of their scores from the mean of path_score, path_prob, path_entropy, path_reward
    pred_paths_revised = {uid: {} for uid in test_labels}
    for uid in pred_paths_revised:
        for pid, paths_info in pred_paths[uid].items():
            if pid not in pred_paths_revised[uid]:
                pred_paths_revised[uid][pid] = []
            for idx, path_info in enumerate(paths_info):
                path_score_diff_user_mean = path_info[0] - user_path_mean_stats[uid][0]
                path_prob_diff_user_mean = path_info[1] - user_path_mean_stats[uid][1]
                path_entropy_diff_user_mean = path_info[2] - user_path_mean_stats[uid][2]
                path_rewards_diff_user_mean = path_info[3] - user_path_mean_stats[uid][3]
                pred_paths_revised[uid][pid].append((path_info[0], path_info[1], path_info[2], path_info[3], path_info[4], path_score_diff_user_mean, path_prob_diff_user_mean, path_entropy_diff_user_mean, path_rewards_diff_user_mean))

    '''if is_debug == 1:
        for uid in pred_paths_revised:
            print('len(pred_paths_revised): ', len(pred_paths_revised[uid].values()))
            print('pred_paths_revised: ', pred_paths_revised[uid].values())'''

    # 2) Path Prioritization - Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for userid in pred_paths_revised:
        train_pids = set(train_labels[userid])
        best_pred_paths[userid] = []
        for pid in pred_paths_revised[userid]:
            if pid in train_pids:
                continue
            if len(pred_paths_revised[userid][pid]) > 0:
                # Get the path with highest probability
                if args.MES_score_option == 0: # Baseline approach
                    # Baseline approach - without explainability score applied
                    sorted_path = pred_paths_revised[userid][pid]
                else:
                    # Path Prioritization - Through Explainability scoring mechanism
                    sorted_path = sorted(pred_paths_revised[userid][pid], key=lambda x: (get_explainability_score(x, args), x[1], x[3], x[2]), reverse=True)
                best_pred_paths[userid].append(sorted_path[0])
    '''if is_debug == 1:
        print('best_pred_paths: ', best_pred_paths)'''

    # 3) Product Prioritization - Compute top 10 recommended products for each user.
    sort_by = 'other' #'reward_per_score'
    #sort_by = 'entropy'
    pred_labels = {}
    pred_labels_path = {}
    pred_labels_details = {}
    pred_labels_details_extended = {}
    sorted_path = {}
    for uid in best_pred_paths:
        if args.PAS_score_option == 0:  # Baseline approach
            sorted_path[uid] = best_pred_paths[uid]
        elif args.PAS_score_option == 1:  # PAS (Product Affinity Score)
            sorted_path[uid] = sorted(best_pred_paths[uid], key=lambda x: (get_product_prioritisation_score(x, args)), reverse = True)
        elif args.PAS_score_option == 2:  # score
            sorted_path[uid] = sorted(best_pred_paths[uid], key=lambda x: (x[0]), reverse=True)
        elif args.PAS_score_option == 3:  # prob
            sorted_path[uid] = sorted(best_pred_paths[uid], key=lambda x: ((x[1] + x[2]), x[0]), reverse=True)
        elif args.PAS_score_option == 4:  # entropy
            sorted_path[uid] = sorted(best_pred_paths[uid], key=lambda x: (x[2]), reverse=True)
        elif args.PAS_score_option == 5:  # reward
            sorted_path[uid] = sorted(best_pred_paths[uid], key=lambda x: (x[3]), reverse=True)

        '''if is_debug == 1:
            print('sorted_path :', sorted_path)'''
        
        top10_pids = [p[-1][2] for _, _, _, _, p, _, _, _, _ in sorted_path[uid][:10]]  # from largest to smallest
        top10_pids_path = [(p[-1][2], str(p[0][1]) + ' ' + str(p[0][2]) + ' has ' + str(p[1][0]) + ' ' + str(p[1][1]) + ' ' + str(p[1][2]) + ' which was ' + str(p[2][0]) + ' by ' + str(p[2][1]) + ' ' + str(p[2][2]) + ' who ' + str(p[3][0]) + ' ' + str(p[3][1]) + ' ' + str(p[3][2])) for _, _, _, _, p, _, _, _, _ in sorted_path[uid] if p[-1][2] in top10_pids]  # from largest to smallest
        top10_pids_details = [(p) for p in sorted_path[uid][:10]]  # from largest to smallest
        #print('scores[uid] :', uid, np.argsort(scores[uid]))

        # add up to 10 pids if not enough
        if args.add_products and len(top10_pids) < 10:
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in top10_pids:
                    continue
                top10_pids.append(cand_pid)
                if len(top10_pids) >= 10:
                    break
        # end of add

        pred_labels[uid] = top10_pids[::1]  # change order to from smallest to largest!
        pred_labels_path[uid] = top10_pids_path[::1]  # change order to from smallest to largest!
        pred_labels_details[uid] = top10_pids_details[::1]  # change order to from smallest to largest!
        pred_labels_details_extended[uid] = [(p[4][-1][2], str(p[4][0][1]) + ' ' + str(p[4][0][2]) + ' has ' + str(p[4][1][0]) + ' ' + str(p[4][1][1]) + ' ' + str(p[4][1][2]) + ' which was ' + str(p[4][2][0]) + ' by ' + str(p[4][2][1]) + ' ' + str(p[4][2][2]) + ' who ' + str(p[4][3][0]) + ' ' + str(p[4][3][1]) + ' ' + str(p[4][3][2]), get_explainability_score(p, args), get_product_prioritisation_score(p, args), p) for p in pred_labels_details[uid]]  # change order to from smallest to largest!
        

    pred_labels_1 = sorted(pred_labels)
    test_labels_1 = sorted(test_labels)
    if is_debug == 1:
        for uid in test_labels_1:
            print('test_labels[uid] :', uid, test_labels[uid])
            print('pred_labels[uid] :', uid, pred_labels[uid])
            print('Path Traversed: User: ', uid, sorted_path[uid])
        for uid in train_labels:
            print('train_labels[uid] :', uid, train_labels[uid])
        for uid in test_labels_1:
            print('test_labels[uid] :', uid, test_labels[uid])
        for uid in pred_labels:
            print('pred_labels[uid] :', uid, pred_labels[uid])

    if is_debug == 1:
        for i in pred_labels:
            for j in (range(len(pred_labels[i]))):
                print('i: j :', i, j)
                print('pred_labels[uid] :', i, pred_labels[i][j])
                print('pred_labels_path[uid] :', i, pred_labels_path[i][j])
                print('pred_labels_details[uid] :', i, pred_labels_details[i][j])
                get_explainability_score(pred_labels_details[i][j], args)
    # Model Evaluation
    #print('Count Pred_labels & test_labels: ', len(pred_labels), len(test_labels))
    ndcg, recall, hit_ratio, precision, invalid_users = evaluate(pred_labels, test_labels, args)
    logger.info(
        'model epoch={:d}'.format(epoch) +
        ' | count (users)={}'.format(len(pred_labels)) +
        ' | ndcg={:.5f}'.format(ndcg) +
        ' | recall={:.5f}'.format(recall) +
        ' | hit_ratio={:.5f}'.format(hit_ratio) +
        ' | precision={:.5f}'.format(precision) +
        ' | invalid_users={:.5f}'.format(invalid_users) +
        ' | execution_timestamp={}'.format(datetime.now())
    )
    return pred_labels, pred_labels_path, pred_labels_details_extended, ndcg, recall, hit_ratio, precision, invalid_users


def test(args, logger):
    start_epoch = 1
    # Parameters created for resumption of run from the last failures
    file_type = r'/*.pkl'
    latest_checkpoint_file = get_latest_file(args.output_dir, file_type)

    # Resume the run from the last failure / saved checkpoint state
    if args.is_only_run_specific_epoch == 0 and args.is_resume_from_checkpoint == 1 and latest_checkpoint_file is not None:
        print('latest_checkpoint_file: ', latest_checkpoint_file)
        start_epoch = int(str.split(latest_checkpoint_file[:-4], '_')[-1]) + 1
    elif args.is_only_run_specific_epoch == 1:
        print('is_only_run_specific_epoch: {} , args.epochs: {}'.format(args.is_only_run_specific_epoch, args.epochs))
        start_epoch = args.epochs
    print('start_epoch: ', start_epoch)

    # Iterate for number of epochs
    for epoch in range(start_epoch, args.epochs + 1):
        policy_file = TMP_DIR[args.dataset] + '/' + args.source_name + '/' + args.checkpoint_folder + '/policy_model_epoch_{}.ckpt'.format(epoch)
        #path_file = args.output_dir + '/policy_paths_epoch{}_{}.pkl'.format(args.epochs, args.run_number)
        path_file = args.output_dir + '/policy_paths_epoch_{}.pkl'.format(epoch)
        is_debug = args.debug
        if is_debug == 1:
            print('policy_file : ', policy_file)
            print('path_file : ', path_file)
            print('args :', args)

        train_labels = load_labels(args.dataset, 'train')
        test_labels = load_labels(args.dataset, 'test')
        #train_labels = {key: value for key, value in train_labels.items() if key in range(20000, 22363, 1)}
        #test_labels = {key: value for key, value in test_labels.items() if key in range(20000, 22363, 1)}
        if args.users is not None:
            train_labels = {key: value for key, value in train_labels.items() if key == args.users}
            test_labels = {key: value for key, value in test_labels.items() if key == args.users}
        if is_debug == 1:
            print('train_labels: ', train_labels)
            print('test_labels: ', test_labels)

        if args.run_path:
            predict_paths(policy_file, path_file, test_labels, args)
        if args.run_eval:
            pred_labels, pred_labels_path, pred_labels_details, ndcg, recall, hit_ratio, precision, invalid_users = evaluate_paths(path_file, train_labels, test_labels, args, epoch, logger)

        return pred_labels, pred_labels_path, pred_labels_details, ndcg, recall, hit_ratio, precision, invalid_users

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--source_name', type=str, default='train_RL_agent', help='directory name.')
    parser.add_argument('--output_folder', type=str, default='test_RL_agent', help='directory name.')
    parser.add_argument('--users', type=int, default=None, help='user list')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[10, 10, 12], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--debug', type=int, nargs='*', default=0, help='number of samples')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--is_resume_from_checkpoint', type=int, default=0, help='Flag for resuming from last checkpoint')
    parser.add_argument('--logging_mode', type=str, default='a', help='logging mode')
    parser.add_argument('--log_file_name', type=str, default='test_agent_log', help='logging mode')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoint', help='Checkpoint folder location')
    parser.add_argument('--MES_score_option', type=int, default=1, help='Choose 0 for [Baseline], Choose 1 for [MES (Rewards Gain * Entropy Gain)], 2 for Only [Rewards Gain], 3 for Only [Entropy Gain], 4 for Only [Probs Gain], 5 for [Entopy Gain * Probs Gain], 6 for [Rewards Gain * Probs Gain], 7 for [Rewards Gain * Entopy Gain * Probs Gain], 8 for [Rewards Gain + Entopy Gain + Probs Gain]')
    parser.add_argument('--PAS_score_option', type=int, default=1, help='Choose 0 for [Baseline], Choose 1 for [PPS ()], 2 for Only [Score], 3 for Only [Prob], 4 for Only [Entropy], 5 for [Reward]')
    parser.add_argument('--run_number', type=int, default='1', help='logging mode')
    parser.add_argument('--is_only_run_specific_epoch', type=int, default=1, help='is_only_run_specific_epoch')
    args = parser.parse_args()

    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cpu')
    if args.gpu == 1:
        if torch.cuda.is_available():
            args.device = torch.device('cuda:0')
    print('args.device: ', args.device)

    is_debug = args.debug
    args.output_dir = TMP_DIR[args.dataset] + '/' + args.output_folder
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logger = get_logger(args.output_dir + '/' + args.log_file_name + '.txt', mode=args.logging_mode)
    logger.info(args)

    set_random_seed(args.seed)
    pred_labels, pred_labels_path, pred_labels_details, ndcg, recall, hit_ratio, precision, invalid_users = test(args, logger)
    
