from __future__ import absolute_import, division, print_function

import os
import sys
from tqdm import tqdm
import pickle
import random
import torch
from datetime import datetime

import knowledge_graph
from knowledge_graph import *
from utils import *


class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')


class BatchKGEnvironment(object):
    def __init__(self, dataset_str, max_acts, max_path_len=3, state_history=1):
        self.dataset = dataset_str
        self.max_acts = max_acts
        self.act_dim = max_acts + 1  # Add self-loop action, whose act_idx is always 0.
        self.max_num_nodes = max_path_len + 1  # max number of hops (= #nodes - 1)
        self.kg = load_kg(dataset_str)
        self.embeds = load_embed(dataset_str)
        self.embed_size = self.embeds[USER].shape[1]
        self.embeds[SELF_LOOP] = (np.zeros(self.embed_size), 0.0)
        self.state_gen = KGState(self.embed_size, history_len=state_history)
        self.state_dim = self.state_gen.dim

        # Compute user-product scores for scaling.
        u_p_scores = np.dot(self.embeds[USER] + self.embeds[PURCHASE][0], self.embeds[PRODUCT].T)
        self.u_p_scales = np.max(u_p_scores, axis=1)

        # Compute path patterns
        self.patterns = []
        for pattern_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
            pattern = PATH_PATTERN[pattern_id]
            pattern = [SELF_LOOP] + [v[0] for v in pattern[1:]]  # pattern contains all relations
            #if pattern_id == 1:
            #    pattern.append(SELF_LOOP)
            self.patterns.append(tuple(pattern))

        # Following is current episode information.
        self._batch_path = None  # list of tuples of (relation, node_type, node_id)
        self._batch_curr_actions = None  # save current valid actions
        self._batch_curr_state = None
        self._batch_curr_reward = None
        self._batch_curr_actions_actual = None  # current valid actual actions leads to target label
        self._batch_curr_actions_actual_idx = None  # current valid actual actions indexes in batch_curr_actions
        self._batch_rewards = []
        self._batch_curr_further_processing = []
        self._batch_action_dict = {}
        # Here only use 1 'done' indicator, since all paths have same length and will finish at the same time.
        self._done = False

    def _has_pattern(self, path, is_debug=0):
        pattern = tuple([v[0] for v in path])
        if is_debug == 1:
            print('pattern : within HasPattern : ', pattern)
            print('self.patterns: ', self.patterns)
        x = pattern in self.patterns
        if is_debug == 1:
            print('pattern in self.patterns: ', str(x))
        return pattern in self.patterns

    def _batch_has_pattern(self, batch_path, is_debug=0):
        # has_pattern = [self._has_pattern(path) for path in batch_path]
        # print('has_pattern : ', str(has_pattern))
        return [self._has_pattern(path, is_debug) for path in batch_path]

    def _further_processing(self, path, reward, is_debug=0):
        if len(path) > 4 or reward <= 0:
            flag = 0
        else:
            flag = 1
        if is_debug == 1:
            print('further processing: ', flag)
        return flag

    def _batch_further_processing(self, batch_path, batch_rewards, is_debug=0):
        return {path[0][2]: self._further_processing(path, batch_rewards[batch_path.index(path)], is_debug) for path in batch_path}

    def _batch_get_idx_curr_actions_actual(self, batch_curr_actions, batch_actions_actual, is_debug=0):
        # Get the batch actions indexes followed by the agent to reach the target label
        idx_batch_curr_actions_actual = [[batch_curr_actions[key].index(item) for item in value if item in batch_actions_actual[key]] for key, value in batch_curr_actions.items()]
        if is_debug == 1:
            print('idx_batch_curr_actions_actual: ', idx_batch_curr_actions_actual)
        return idx_batch_curr_actions_actual

    def _get_actions(self, path, done, target_product=0, is_train=0, is_debug=0):
        if is_debug == 1:
            print('Start: _get_actions')
            print('path:', path, 'target_product: ', target_product)
        """Compute actions for current node."""
        last_relation, curr_node_type, curr_node_id = path[-1]
        actions = [(SELF_LOOP, curr_node_id)]  # self-loop must be included.
        # (1) If game is finished, only return self-loop action.
        if done:
            return actions
        # (2) Get all possible edges from original knowledge graph. # [CAVEAT] Must remove visited or common nodes!
        relations_nodes = self.kg(curr_node_type, curr_node_id)
        relations_nodes_history = relations_nodes.copy()
        if is_debug == 1:
            print('relations_nodes:pre:', relations_nodes)
        candidate_acts = []  # list of tuples of (relation, node_type, node_id)
        visited_nodes = set([(v[1], v[2]) for v in path])
        if is_debug == 1:
            print('visited_nodes:', visited_nodes)
        for r in relations_nodes_history:
            next_node_type = KG_RELATION[curr_node_type][r]
            next_node_ids = relations_nodes_history[r]
            # Remove most common nodes
            if next_node_type == 'category':
                visited_nodes.update([('category', 0)])
            if last_relation == SELF_LOOP and is_train == 1 and next_node_type == PRODUCT and target_product != 0:
                visited_nodes.update([('product', target_product)])
            next_node_ids = [n for n in next_node_ids if (next_node_type, n) not in visited_nodes]  # filter
            if is_debug == 1:
                print('r:', r, 'next_node_type:', next_node_type, 'next_node_ids:', next_node_ids)
            candidate_acts.extend(zip([r] * len(next_node_ids), next_node_ids))
        # (3) If candidate action set is empty, only return self-loop action.
        if len(candidate_acts) == 0:
            return actions
        # (4) If number of available actions is smaller than max_acts, return action sets.
        if len(candidate_acts) <= self.max_acts:
            #candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
            actions.extend(candidate_acts)
            if is_debug == 1:
                print('len(candidate_acts):', len(candidate_acts))
                print('actions:', actions)
            return actions
        # (5) If there are too many actions, do some deterministic trimming here!
        user_embed = self.embeds[USER][path[0][-1]]
        scores = []
        for r, next_node_id in candidate_acts:
            next_node_type = KG_RELATION[curr_node_type][r]
            if next_node_type == USER:
                src_embed = user_embed
            elif next_node_type == PRODUCT:
                src_embed = user_embed + self.embeds[PURCHASE][0]
            elif next_node_type == WORD:
                src_embed = user_embed + self.embeds[MENTION][0]
            else:  # BRAND, CATEGORY, RELATED_PRODUCT
                src_embed = user_embed + self.embeds[PURCHASE][0] + self.embeds[r][0]
            score = np.matmul(src_embed, self.embeds[next_node_type][next_node_id])
            # This trimming may filter out target products!
            # Manually set the score of target products a very large number.
            if is_train == 1 and next_node_type == PRODUCT and next_node_id == target_product: #self._target_pids:
                score += 99999.0
            scores.append(score)
        candidate_idxs = np.argsort(scores)[-self.max_acts:]  # choose actions with larger scores
        #candidate_acts = sorted([candidate_acts[i] for i in candidate_idxs], key=lambda x: (x[0], x[1])) # Existing code block
        candidate_acts = [candidate_acts[i] for i in candidate_idxs]  # New code block
        #candidate_acts = sorted(candidate_acts, key=lambda x: (x[0], x[1]))
        #print('candidate_acts: Post: ', candidate_acts)
        actions.extend(candidate_acts)
        if is_debug == 1:
            print('too many actions')
            print('actions:', actions)
        return actions

    def _batch_get_actions(self, batch_path, done, uids, batch_curr_further_processing=None, is_train=0, is_debug=0):
        if is_debug == 1:
            print('Start: _batch_get_actions')
            for path in batch_path:
                print('is_train == 1 and uids[path[0][2]] != -9999 and batch_curr_further_processing[path[0][2]]: ', is_train, uids[path[0][2]], batch_curr_further_processing[path[0][2]])
        if is_train == 1:
            batch_get_actions = [self._get_actions(path, done if ((is_train == 1 and uids[path[0][2]] != -9999 and batch_curr_further_processing[path[0][2]] == 1) or (is_train == 0)) else True, uids[path[0][2]], is_train, is_debug) for path in batch_path]
        else:
            batch_get_actions = [self._get_actions(path, done, 0, is_train, is_debug) for path in batch_path]
        return batch_get_actions

    def _get_state(self, path):
        """Return state of numpy vector: [user_embed, curr_node_embed]."""
        user_embed = self.embeds[USER][path[0][-1]]
        zero_embed = np.zeros(self.embed_size)
        if len(path) == 1:  # initial state
            state = self.state_gen(user_embed, user_embed, zero_embed, zero_embed, zero_embed, zero_embed)
            return state
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation_embed]."""
        older_relation, last_node_type, last_node_id = path[-2]
        last_relation, curr_node_type, curr_node_id = path[-1]
        curr_node_embed = self.embeds[curr_node_type][curr_node_id]
        last_node_embed = self.embeds[last_node_type][last_node_id]
        last_relation_embed, _ = self.embeds[last_relation]  # this can be self-loop!
        if len(path) == 2:
            state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, zero_embed, zero_embed)
            return state
        """Return state of numpy vector: [user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed]."""
        _, older_node_type, older_node_id = path[-3]
        older_node_embed = self.embeds[older_node_type][older_node_id]
        older_relation_embed, _ = self.embeds[older_relation]
        state = self.state_gen(user_embed, curr_node_embed, last_node_embed, last_relation_embed, older_node_embed, older_relation_embed)
        return state

    def _batch_get_state(self, batch_path):
        batch_state = [self._get_state(path) for path in batch_path]
        return np.vstack(batch_state)  # [bs, dim]

    def _get_reward(self, path, act_idx=None, act_actual_idx=None, target_product=0, is_train=0, is_debug=0):
        target_score = 0.0
        if is_debug == 1:
            print('Start: _get_reward')
            print('path:', path, 'target_product: ', target_product)
            print('act_idx: ', act_idx)
            print('act_actual_idx: ', act_actual_idx)

        # If it is initial state or 1-hop search, reward is 0.
        if is_train == 0 and len(path) <= 2:
            return 0.0
        # Evaluate path pattern
        is_pattern = self._has_pattern(path, is_debug)
        # Return if not following path pattern
        if len(path) >= 3 and not is_pattern:
            target_score -= (200 * len(path))
            return target_score

        _, curr_node_type, curr_node_id = path[-1]
        if is_train == 1 and (act_idx in act_actual_idx):
            target_score += (100 * len(path))
        #else:
        #    return 0.0
        if len(path) >= 3 and curr_node_type == PRODUCT and is_pattern:
            # Give soft reward for other reached products.
            uid = path[0][-1]
            u_vec = self.embeds[USER][uid] + self.embeds[PURCHASE][0]
            p_vec = self.embeds[PRODUCT][curr_node_id]
            score = np.dot(u_vec, p_vec) / self.u_p_scales[uid]
            score *= (10000/len(path))  # added
            target_score += max(score, 0.0)
            if is_train == 1 and curr_node_id == target_product:
                target_score += 99999.0
            if is_debug == 1:
                print('curr_node_type : ', curr_node_type, 'target_score: ', target_score)  # added
        elif len(path) == 3 and is_pattern:
            # Give soft reward for other reached products.
            uid = path[0][-1]
            u_vec = self.embeds[USER][uid]
            p_vec = self.embeds[curr_node_type][curr_node_id]
            #score = np.dot(u_vec, p_vec)  # / self.u_p_scales[uid]
            score = np.matmul(u_vec, p_vec)
            score *= (100 * len(path))  # added
            target_score += max(score, 0.0)
            if is_debug == 1:
                print('curr_node_type : ', curr_node_type, 'target_score: ', target_score)  # added

        if is_debug == 1:
            print('target_score : ', target_score)

        return target_score

    def _batch_get_reward(self, batch_path, uids, batch_act_idx=None, batch_act_actual_idx=None, is_train=0, is_debug=0):
        if is_debug == 1:
            print('Start: _batch_get_reward')
            print('batch_path :', batch_path)
            print('uids: ', uids)
            print('batch_act_idx: ', batch_act_idx)
            print('batch_act_actual_idx ', batch_act_actual_idx)
        if is_train == 1:
            batch_reward = [self._get_reward(path, batch_act_idx[batch_path.index(path)], batch_act_actual_idx[batch_path.index(path)], uids[path[0][2]], is_train=is_train, is_debug=is_debug) if ((is_train == 1 and uids[path[0][2]] != -9999) or (is_train == 0)) else 0 for path in batch_path]
        else:
            batch_reward = [self._get_reward(path, None, None, -9999, is_train=0, is_debug=is_debug) for path in batch_path]
        return np.array(batch_reward)

    def _is_done(self):
        """Episode ends only if max path length is reached."""
        return self._done or max(len(path) for path in self._batch_path) >= self.max_num_nodes or (all(reward <= 0 for reward in self._batch_curr_reward))

    def reset(self, uids=None, is_train=0, is_debug=0):
        if uids is None:
            all_uids = list(self.kg(USER).keys())
            uids = [random.choice(all_uids)]
        # each element is a tuple of (relation, entity_type, entity_id)
        self._batch_path = [[(SELF_LOOP, USER, uid)] for uid in uids]
        self._batch_curr_actions_actual_idx = None
        self._done = False
        self._batch_curr_further_processing = {path[0][2]: 1 for path in self._batch_path}
        self._batch_rewards = [[0] for i in range(len(self._batch_path))]
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done, uids, self._batch_curr_further_processing, is_train=is_train, is_debug=is_debug)
        if is_train == 1:
            self._batch_action_dict = {self._batch_path[i][0][2]: self._batch_curr_actions[i] for i in range(len(self._batch_curr_actions))}
            self._batch_curr_actions_actual = knowledge_graph.batch_get_user_product_path_actions_actual(self.dataset, self.kg, uids, self._batch_path, self._batch_curr_further_processing, is_train=is_train, is_debug=is_debug)
            self._batch_curr_actions_actual_idx = self._batch_get_idx_curr_actions_actual(self._batch_action_dict, self._batch_curr_actions_actual, is_debug=is_debug)
        #if is_train == 0:
        #    self._batch_curr_reward = self._batch_get_reward(self._batch_path, uids, is_train=is_train, is_debug=is_debug)
        if is_debug == 1:
            print('Start: RESET Step')
            print('uids: ', uids)
            print('self._batch_path: ', self._batch_path)
            print('self._batch_curr_actions: ', self._batch_curr_actions)
            if is_train == 1:
                print('batch_action_dict: ', self._batch_action_dict)
                print('self._batch_curr_actions_actual: ', self._batch_curr_actions_actual)
                print('self._batch_curr_actions_actual_idx: ', self._batch_curr_actions_actual_idx)
            print('End: RESET Step')
        return self._batch_curr_state, self._batch_path, self._batch_curr_actions, self._batch_curr_reward, self._batch_curr_actions_actual_idx, self._done

    def batch_step(self, batch_act_idx, uids=None, is_train=0, is_debug=0):
        """
        Args:
            batch_act_idx: list of integers.
        Returns:
            batch_next_state: numpy array of size [bs, state_dim].
            batch_reward: numpy array of size [bs].
            done: True/False
        """
        assert len(batch_act_idx) == len(self._batch_path)
        # Evaluate rewards for the model based on the actual actions and selected actions
        if is_train == 1:
            self._batch_curr_reward = self._batch_get_reward(self._batch_path, uids, batch_act_idx, self._batch_curr_actions_actual_idx, is_train=is_train, is_debug=is_debug)
            # Evaluate the further processing decision
            self._batch_curr_further_processing = self._batch_further_processing(self._batch_path, self._batch_curr_reward, is_debug=is_debug)
        # Execute batch actions.
        for i in range(len(batch_act_idx)):
            if (is_train == 1 and self._batch_curr_further_processing[self._batch_path[i][0][2]] == 1) or is_train == 0:
                act_idx = batch_act_idx[i]
                _, curr_node_type, curr_node_id = self._batch_path[i][-1]
                relation, next_node_id = self._batch_curr_actions[i][act_idx]
                if relation == SELF_LOOP:
                    next_node_type = curr_node_type
                else:
                    next_node_type = KG_RELATION[curr_node_type][relation]
                self._batch_path[i].append((relation, next_node_type, next_node_id))
                self._batch_rewards[i].append(self._batch_curr_reward[i])
        self._done = self._is_done()  # must run before get actions, etc.
        self._batch_curr_state = self._batch_get_state(self._batch_path)
        self._batch_curr_actions = self._batch_get_actions(self._batch_path, self._done, uids, self._batch_curr_further_processing, is_train=is_train, is_debug=is_debug)
        if is_train == 1:
            self._batch_action_dict = {self._batch_path[i][0][2]: self._batch_curr_actions[i] for i in range(len(self._batch_curr_actions))}
            self._batch_curr_actions_actual = knowledge_graph.batch_get_user_product_path_actions_actual(self.dataset, self.kg, uids, self._batch_path, self._batch_curr_further_processing, is_train=is_train, is_debug=0)
            self._batch_curr_actions_actual_idx = self._batch_get_idx_curr_actions_actual(self._batch_action_dict, self._batch_curr_actions_actual, is_debug=is_debug)
        if is_debug == 1:
            print('Start: batch_step')
            print('uids: ', uids)
            print('batch_act_idx: ', batch_act_idx)
            print('self._batch_curr_reward: ', self._batch_curr_reward)
            if is_train == 1:
                print('self._batch_curr_further_processing: ', self._batch_curr_further_processing)
                print('self._batch_curr_actions: ', self._batch_curr_actions)
                print('self._batch_action_dict: ', self._batch_action_dict)
                print('self._batch_curr_actions_actual: ', self._batch_curr_actions_actual)
                print('self._batch_curr_actions_actual_idx: ', self._batch_curr_actions_actual_idx)
            print('self._done: ', self._done)
            print('End: batch_step')
        # adjust the reward for the last lap
        if is_train == 1 and self._done and any([self._batch_curr_further_processing[uid] == 1 for uid in self._batch_curr_further_processing]):
            if is_debug == 1:
                print('Start: adjust the reward for the last lap')
            self._batch_curr_reward = self._batch_get_reward(self._batch_path, uids, batch_act_idx, self._batch_curr_actions_actual_idx, is_train=is_train, is_debug=is_debug)
        elif is_train == 0:
            self._batch_curr_reward = self._batch_get_reward(self._batch_path, uids, is_train=is_train, is_debug=is_debug)
        return self._batch_curr_state, self._batch_path, self._batch_curr_actions, self._batch_curr_reward, self._batch_rewards, self._batch_curr_actions_actual_idx, self._done

    def batch_action_mask(self, dropout=0.0):
        """Return action masks of size [bs, act_dim]."""
        batch_mask = []
        # For block related to self._batch_curr_actions
        for actions in self._batch_curr_actions:
            act_idxs = list(range(len(actions)))
            tmp = act_idxs[1:]
            #tmp = act_idxs  #[1:]
            if dropout > 0 and len(act_idxs) >= 5:
                keep_size = int(len(act_idxs[1:]) * (1.0 - dropout))
                tmp = np.random.choice(tmp, keep_size, replace=False).tolist()
            #act_idxs = [act_idxs[0]] + tmp
            act_idxs = tmp
            #print('act_idxs: batch_action_mask: ', act_idxs)
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        '''
        for act_idxs in self._batch_curr_actions_actual_idx:
            act_mask = np.zeros(self.act_dim, dtype=np.uint8)
            act_mask[act_idxs] = 1
            batch_mask.append(act_mask)
        '''
        return np.vstack(batch_mask)

    def print_path(self):
        for path in self._batch_path:
            msg = 'Path: {}({})'.format(path[0][1], path[0][2])
            for node in path[1:]:
                msg += ' =={}=> {}({})'.format(node[0], node[1], node[2])
            print(msg)

