from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
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

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from train_agent_test import ActorCritic
from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#### Function to get the quanitification of explainability
def evaluate_explainability(pred_labels_details):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.

        Quantitative evaluation of explainability
        R - the number of rules outputted by the explanation
        S - The average score of the path traversal for the recommended item
        P - The average probability of the path traversal for the recommended item
        Rw- The average reward of the path traversal for the recommended item

        Formula = (S + P + Rw)/ ((MAX range(S) + MAX range(P) + MAX range(Rw))) * R
    """
    invalid_users = []
    # Extract the key metrics from the prediction label details
    #for i in range(len(pred_labels_details)):
    #print('i :', i)
    # Score of the prediction label
    pred_score = pred_labels_details[0]
    # Probability of the prediction label
    pred_probs = pred_labels_details[1]
    # Entropy of the prediction label
    pred_entropy = pred_labels_details[2]
    # Rewards of the prediction label
    pred_reward = pred_labels_details[3]
    # Path traverses by the user to reach to the prediction label
    pred_path = pred_labels_details[4]
    print('pred_score={} |  pred_probs={} | pred_entropy={} | pred_reward={} | | pred_path={} | len(pred_path)={}'.format(
        pred_score, pred_probs, pred_entropy, pred_reward, pred_path, len(pred_path)))

    # Compute metrics for Explainability
    # The explainability of a prediction means that how the prediction has been derived by the AI system.
    # If a product has been recommended to a user, then it could be possible that there could be many ways that the user might reached to that product.
    #
    print(((pred_score + pred_reward + pred_probs) / (len(pred_path))) * pred_entropy)

#def fn_get_master_data(id, )

def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
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
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
        avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    #print('state_pool:', state_pool)
    path_pool = env._batch_path  # list of list, size=bs
    print('path_pool:', path_pool)
    probs_pool = [[] for _ in uids]
    print('probs_pool:', probs_pool)
    # Rewards for the paths
    rewards_pool = [[] for _ in uids]

    model.eval()
    for hop in range(3):
        print('hop:',hop)
        state_tensor = torch.FloatTensor(state_pool).to(device)
        #print('state_tensor:', state_tensor)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        #print('acts_pool:', acts_pool)
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        #print('actmask_pool:', actmask_pool)
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        #print('actmask_tensor:', actmask_tensor)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        print('probs: Earlier: ', probs)
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        #print('probs: After: ', probs)
        print('topk[hop]:', topk[hop])
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        #print('topk_probs:', topk_probs, 'topk_idxs:', topk_idxs)

        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()
        print('topk_probs:', topk_probs, 'topk_idxs:', topk_idxs)
        print('topk_idxs.shape', topk_idxs.shape)

        new_path_pool, new_probs_pool, new_rewards_pool, new_scores_pool = [], [], [], []
        for row in range(topk_idxs.shape[0]):
            print('row', row)
            path = path_pool[row]
            print('path:', path)
            probs = probs_pool[row]
            print('probs:', probs)
            #print('topk_idxs[row], topk_probs[row]:', topk_idxs[row], topk_probs[row])
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                print('idx, p:', idx, p)
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                print('acts_pool[row]:', acts_pool[row])
                print('acts_pool[row][idx]:', acts_pool[row][idx])
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                print('relation:', relation)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                print('next_node_type:', next_node_type)
                new_path = path + [(relation, next_node_type, next_node_id)]
                print('new_path:', new_path)
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        print('new_path_pool:', new_path_pool)
        print('new_probs_pool:', new_probs_pool)
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, test_labels, args):
    print('Predicting paths...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = torch.load(policy_file)
    print('env.state_dim : ', env.state_dim)
    print('env.act_dim : ', env.act_dim)
    print('args.gamma : ', args.gamma)
    print('args.hidden : ', args.hidden)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    #test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())
    print('length - test_uids : ', len(test_uids))
    print('test_uids : ', test_uids)

    batch_size = 16
    start_idx = 0
    all_paths, all_probs, all_entropy, all_rewards = [], [], [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        print(batch_uids)
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
        #print('batch_uids : ', batch_uids, 'paths :', paths, 'probs : ', probs)
        probs_tensor = torch.FloatTensor(probs).to(args.device)
        all_paths.extend(paths)
        all_probs.extend(probs)
        # Entropy calculation
        entropy = Categorical(probs_tensor).entropy()
        entropy = entropy.detach().cpu().numpy()
        all_entropy.extend(entropy)
        # Rewards
        rewards = env._batch_get_reward(paths)
        all_rewards.extend(rewards)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs, 'entropy': all_entropy, 'rewards': all_rewards}

    print('paths : ', all_paths)
    print('max(paths) : ', max(all_paths), 'len(paths):', len(all_paths))

    print('probs : ', all_probs)
    print('max(probs) : ', max(all_probs), 'len(probs):', len(all_probs))

    print('entropy : ', all_entropy)
    print('max(entropy) : ', max(all_entropy), 'len(entropy):', len(all_entropy))

    print('rewards : ', all_rewards)
    print('max(rewards) : ', max(all_rewards), 'len(rewards):', len(all_rewards))

    print('predicts : ', predicts)
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(path_file, train_labels, test_labels):
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
        path_prob = reduce(lambda x, y: x * y, probs)
        path_entropy = entropy
        path_reward = reward
        pred_paths[uid][pid].append((path_score, path_prob, path_entropy, path_reward, path))

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            # Get the path with highest probability
            sorted_path1 = sorted(pred_paths[uid][pid], key=lambda x: (x[1]), reverse=True)
            best_pred_paths[uid].append(sorted_path1[0])

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'prob'
    #sort_by = 'entropy'
    pred_labels = {}
    pred_labels_path = {}
    pred_labels_details = {}
    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: ((x[0] + x[3]) * x[2] * x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: ((x[1] + x[2]), x[0]), reverse=True)
        elif sort_by == 'entropy':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[2]), reverse=True)
        elif sort_by == 'reward':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[3], x[0], x[2]), reverse=True)

        #print('sorted_path :', sorted_path)
        top10_pids = [p[-1][2] for _, _, _, _, p in sorted_path[:10]]  # from largest to smallest
        top10_pids_path = [(p[-1][2], str(p[0][1]) + ' ' + str(p[0][2]) + ' has ' + str(p[1][0]) + ' ' + str(p[1][1]) + ' ' + str(p[1][2]) + ' which was ' + str(p[2][0]) + ' by ' + str(p[2][1]) + ' ' + str(p[2][2]) + ' who ' + str(p[3][0]) + ' ' + str(p[3][1]) + ' ' + str(p[3][2])) for _, _, _, _, p in sorted_path[:10]]  # from largest to smallest
        top10_pids_details = [(p) for p in sorted_path[:10]]  # from largest to smallest
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
        #print('pred_labels[uid] :', uid, pred_labels[uid])

    pred_labels_1 = sorted(pred_labels)
    test_labels_1 = sorted(test_labels)
    print('len(pred_labels_1) : ', len(pred_labels_1))
    print('len(test_labels_1) : ', len(test_labels_1))
    #for i in test_labels_1[:len(test_labels_1) - 5:-1]:
    for i in train_labels:
        print('train_labels[uid] :', i, train_labels[i])
    for i in test_labels_1:
        print('test_labels[uid] :', i, test_labels[i])
    for i in pred_labels:
        print('pred_labels[uid] :', i, pred_labels[i])

    for i in pred_labels_1[:len(pred_labels_1) - 5:-1]:
        for j in (range(len(pred_labels[i]))):
            print('i: j :', i, j)
            print('pred_labels[uid] :', i, pred_labels[i][j])
            print('pred_labels_path[uid] :', i, pred_labels_path[i][j])
            print('pred_labels_details[uid] :', i, pred_labels_details[i][j])
            evaluate_explainability(pred_labels_details[i][j])
    print('Shape: pred_paths: ', len(pred_paths), type(pred_paths))
    print('Shape: pred_paths: ', len(pred_paths[22341]), len(pred_paths[22342]))
    print('User Id: 22341 : ', pred_paths[22341])
    print('User Id: 22342 : ', pred_paths[22342])

    evaluate(pred_labels, test_labels)


def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    #path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)
    path_file = args.log_dir + '/Test_Labels_Prediction.pkl'.format(args.epochs)
    print('policy_file : ', policy_file)
    print('path_file : ', path_file)
    print('args :', args)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')
    #print('test_labels :', dict(sorted(test_labels.items())))
    #print('train_labels :', dict(sorted(train_labels.items())))

#    train_labels = {22341: [10463, 4215, 8177, 591, 12007, 10105, 6316, 331, 1317, 10430, 5635, 6767, 587, 8322, 5395, 6730, 10244, 5213, 7817, 5884, 6451, 8365, 359, 9389, 6629, 10732, 8185, 599, 5334, 11798, 9004, 9116, 3135], 22342: [1885, 8885, 5210, 1768, 2163, 8417, 1153, 5672, 5063, 257, 1442, 1847, 2500, 6131], 22343: [5017, 7666, 9849, 3558, 3320], 22344: [11842, 1885, 6912, 2673, 3746, 2313, 3725, 5867, 1427, 9278], 22345: [8507, 1853, 3128, 2089, 9437], 22346: [11835, 4719, 11629, 1105], 22347: [11364, 2313, 3084, 8386, 7129], 22348: [858, 6258, 7813, 8940, 8268, 6383, 10933], 22349: [5676, 6165, 4435, 6446, 10833], 22350: [8994, 9890, 3361, 8031], 22351: [1680, 4899, 10033, 5997, 1700], 22352: [210, 7221, 8768, 11314], 22353: [2057, 2556, 9797, 2570], 22354: [3243, 10363, 1378, 5440, 2756, 11905], 22355: [5216, 6179, 9415, 954, 3748, 12038, 458], 22356: [614, 8295, 2750, 11923], 22357: [6367, 6443, 8191, 2647], 22358: [11816, 10927, 8308, 2436, 5623, 1856, 2149, 7989, 7773], 22359: [11599, 366, 5561, 1496, 7227, 11310, 5732, 1819, 5984, 2237, 246, 208, 5187, 9463, 5738, 11581, 8147, 2105, 7512, 6252, 9856, 5480, 4139], 22360: [4830, 2120, 11155, 5903, 4858], 22361: [6966, 2204, 9590, 5098, 1929], 22362: [10338, 1220, 4430, 2341, 1757, 5748, 5331, 7746]}
#    test_labels = {22341: [3253, 9439, 11393, 5432, 8616, 3412, 10474, 540, 2823, 8720, 8889, 1812, 9568], 22342: [1117, 9042, 6758, 711, 2478], 22343: [5090], 22344: [7018, 9148, 9122], 22345: [8564], 22346: [2776], 22347: [9148], 22348: [9617, 7304], 22349: [5714, 3107], 22350: [10994], 22351: [6448, 4875], 22352: [10709], 22353: [6009], 22354: [9106, 6290], 22355: [10969, 7902, 4870], 22356: [6955], 22357: [5809], 22358: [5119, 9810, 2367], 22359: [4073, 7963, 851, 651, 11533, 10925, 10460, 9996, 9257], 22360: [561, 792], 22361: [11615, 5506], 22362: [43, 1635, 6384]}
    train_labels = {22341: [10463, 4215, 8177, 591, 12007, 10105, 6316, 331, 1317, 10430, 5635, 6767, 587, 8322, 5395, 6730, 10244, 5213, 7817, 5884, 6451, 8365, 359, 9389, 6629, 10732, 8185, 599, 5334, 11798, 9004, 9116, 3135], 22342: [1885, 8885, 5210, 1768, 2163, 8417, 1153, 5672, 5063, 257, 1442, 1847, 2500, 6131]}
    test_labels = {22341: [3253, 9439, 11393, 5432, 8616, 3412, 10474, 540, 2823, 8720, 8889, 1812, 9568], 22342: [1117, 9042, 6758, 711, 2478]}

    if args.run_path:
        predict_paths(policy_file, path_file, test_labels, args)
    if args.run_eval:
        evaluate_paths(path_file, train_labels, test_labels)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=50, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    test(args)

