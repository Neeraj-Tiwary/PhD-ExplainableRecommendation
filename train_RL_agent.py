from __future__ import absolute_import, division, print_function

# Generic libraries and packages
import sys
import os
import argparse
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# Python script files
from knowledge_graph import *
from kg_env import BatchKGEnvironment
from utils import *

logger = None
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)
        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs, device="cpu", is_debug=0):
        state, act_mask = inputs  # state: [bs, state_dim], act_mask: [bs, act_dim]
        state = state.to(device)
        act_mask = act_mask.to(device)
        x = self.l1(state).to(device)
        x = F.dropout(F.elu(x), p=0.5).to(device)
        x = self.l2(x).to(device)
        x = F.dropout(F.elu(x), p=0.5).to(device)
        actor_logits = self.actor(x).to(device)
        if is_debug == 1:
            print('actor_logits: pre: ', actor_logits)
        # byte tensor
        one_minus_act_mask = (1- act_mask).type(torch.bool).to(device)
        #one_minus_act_mask = 1- act_mask
        actor_logits[one_minus_act_mask] = -999999.0
        if is_debug == 1:
            print('actor_logits: post: ', actor_logits)
        act_probs = F.softmax(actor_logits, dim=-1)  # Tensor of [bs, act_dim]
        if is_debug == 1:
            print('act_probs: forward: ', act_probs)
        state_values = self.critic(x)  # Tensor of [bs, 1]
        return act_probs, state_values

    def select_action(self, batch_states, batch_action_mask, device='cpu', is_debug=0):
        state = torch.FloatTensor(batch_states).to(device)  # Tensor [bs, state_dim]
        act_mask = torch.ByteTensor(batch_action_mask).to(device)  # Tensor of [bs, act_dim]
        probs, value = self((state, act_mask), device)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        if is_debug == 1:
            print('Select_Action: Started...')
            print('probs:', probs)
        m = Categorical(probs)
        acts = m.sample()  # Tensor of [bs, ], requires_grad=False
        # [CAVEAT] If sampled action is out of action_space, choose the first action in action_space.
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0
        if is_debug == 1:
            print('m: ', m)
            print('acts: ', acts)
            print('valid_idx: select_action: ', valid_idx)
            print('log_prob(acts) :', m.log_prob(acts))
            print('acts.cpu().numpy().tolist(): ', acts.cpu().numpy().tolist())
            print('Select_Action: Ended...')
        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self, optimizer, device, ent_weight, batch_size):
        if len(self.rewards) <= 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0
        batch_rewards = np.vstack(self.rewards).T  # numpy array of [bs, #steps]
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0 #torch.zeros([batch_size, ]).to(device)
        critic_loss = 0 #torch.zeros([batch_size, ]).to(device)
        entropy_loss = 0 # torch.zeros([batch_size, ]).to(device)
        for i in range(0, num_steps):
            log_prob, value = self.saved_actions[i]  # log_prob: Tensor of [bs, ], value: Tensor of [bs, 1]
            advantage = batch_rewards[:, i] - value.squeeze(1)  # Tensor of [bs, ]
            #print('advantage: ', advantage)
            #print('log_prob: ', log_prob)
            actor_loss += -log_prob * advantage.detach()  # Tensor of [bs, ]
            critic_loss += advantage.pow(2)  # Tensor of [bs, ]
            entropy_loss += -self.entropy[i]  # Tensor of [bs, ]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        #loss = actor_loss + critic_loss + ent_weight * entropy_loss
        loss = actor_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()


class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()


def train(args):
    # Environment setting up knowledge graph propagation
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    is_debug = args.debug

    # Get the list of users for training the model
    uids = list(env.kg(USER).keys())
    if is_debug == 1:
        uids = [22341, 22342]

    # Get corresponding train labels based on the users in the train dataset
    train_labels = load_labels(args.dataset, 'train')
    train_labels = dict(sorted((key, train_labels[key]) for key in train_labels.keys() if key in uids))
    if is_debug == 1:
        print('train_labels: ', train_labels)

    # Define the ActorCritic model
    dataloader = ACDataLoader(uids, args.batch_size)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
    step = 0
    start_epoch = 1

    # Parameters created for resumption of run from the last failures
    file_type = r'/*.ckpt'
    latest_checkpoint_file = get_latest_file(args.checkpoint_dir, file_type)

    # Resume the run from the last failure / saved checkpoint state
    if args.is_resume_from_checkpoint == 1 and latest_checkpoint_file is not None:
        print('latest_checkpoint_file: ', latest_checkpoint_file)
        latest_checkpoint = load_checkpoint(latest_checkpoint_file)
        model.load_state_dict(latest_checkpoint['model_state_dict'])
        optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
        step = latest_checkpoint['step']
        start_epoch = latest_checkpoint['epoch'] + 1

    # Iterate for number of epochs
    for epoch in range(start_epoch, args.epochs + 1):
        # Start epoch #
        dataloader.reset()
        model.train().to(args.device)
        while dataloader.has_next():
            # Get the batch Users
            batch_uids = dataloader.get_batch()
            # Get the train labels corresponding to batch users
            batch_uids_train_labels = dict(sorted((key, train_labels[key]) for key in train_labels.keys() if key in (batch_uids)))
            batch_uids_train_label_length = dict(sorted((key, len(batch_uids_train_labels[key])) for key in batch_uids_train_labels.keys() if key in (batch_uids)))
            max_length_batch_uids_train_label = max(batch_uids_train_label_length.values())
            if is_debug == 1:
                print('batch_uids_train_label_length: pre: ', batch_uids_train_label_length)
                print('max_length_batch_uids_train_label:', max_length_batch_uids_train_label)

            for key in batch_uids_train_labels.keys():
                if len(batch_uids_train_labels[key]) < max_length_batch_uids_train_label:
                    for i in range(len(batch_uids_train_labels[key]), max_length_batch_uids_train_label):
                        batch_uids_train_labels[key].append(-9999)

            for i in range(max_length_batch_uids_train_label):
                # Start batch episodes #
                batch_uids_train_label = dict(sorted((key, (batch_uids_train_labels[key][i])) for key in batch_uids_train_labels.keys()))
                if is_debug == 1:
                    print('batch_uids_train_label: ', batch_uids_train_label)
                batch_state, batch_path, batch_action, batch_reward, batch_actions_actual_idx, done = env.reset(batch_uids_train_label, is_train=1, is_debug=is_debug)  # numpy array of [bs, state_dim]
                if is_debug == 1:
                    # print('batch_state: ', batch_state)
                    print('reset: batch_path: ', batch_path)
                    #print('reset: batch_action: ', batch_action)
                    print('reset: batch_reward: ', batch_reward)
                    print('reset: batch_actions_actual_idx: ', batch_actions_actual_idx)
                    print('reset: done: ', done)
                done = False
                while not done:
                    batch_act_mask = env.batch_action_mask(dropout=args.act_dropout)  # numpy array of size [bs, act_dim]
                    if is_debug == 1:
                        print('Start: while loop')
                        print('batch_act_mask: ', batch_act_mask)
                    batch_act_idx = model.select_action(batch_state, batch_act_mask, device=args.device, is_debug=is_debug)  # int
                    if is_debug == 1:
                        print('batch_act_idx: ', batch_act_idx)
                    batch_state, batch_path, batch_action, batch_curr_reward, batch_rewards, batch_actions_actual_idx, done = env.batch_step(batch_act_idx, batch_uids_train_label, is_train=1, is_debug=is_debug)
                    model.rewards.append(batch_curr_reward)
                    if is_debug == 1:
                        # print('batch_state: ', batch_state)
                        print('batch_path: ', batch_path)
                        #print('batch_action: ', batch_action)
                        print('batch_curr_reward: ', batch_curr_reward)
                        print('batch_total_rewards: ', batch_rewards)
                        print('batch_actions_actual_idx: ', batch_actions_actual_idx)
                        print('done: ', done)
                        print('model.rewards: ', model.rewards)
                ### End of episodes ###
                    lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(uids) / args.batch_size))
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr

                    if is_debug == 1:
                        print('End of episodes: model.rewards: ', model.rewards)
                        print('lr: ', lr)
                    # Update policy
                    total_rewards.append(np.sum(model.rewards))
                    loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight, args.batch_size)
                    total_losses.append(loss)
                    total_plosses.append(ploss)
                    total_vlosses.append(vloss)
                    total_entropy.append(eloss)
                    step += 1

                    # Report performance
                    if step > 0 and step % args.steps_per_checkpoint == 0:
                        avg_reward = np.mean(total_rewards) / args.batch_size
                        avg_loss = np.mean(total_losses)
                        avg_ploss = np.mean(total_plosses)
                        avg_vloss = np.mean(total_vlosses)
                        avg_entropy = np.mean(total_entropy)
                        total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
                        logger.info(
                            'epoch/step={:d}/{:d}'.format(epoch, step) +
                            ' | loss={:.5f}'.format(avg_loss) +
                            ' | ploss={:.5f}'.format(avg_ploss) +
                            ' | vloss={:.5f}'.format(avg_vloss) +
                            ' | entropy={:.5f}'.format(avg_entropy) +
                            ' | reward={:.5f}'.format(avg_reward))
        ### END of epoch ###
        # Saving the policy epoch file
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }
        policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.checkpoint_dir, epoch)
        logger.info("Save model to " + policy_file)
        save_checkpoint(checkpoint, policy_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=CLOTH, help='One of {beauty, cd, cell, cloth}')
    parser.add_argument('--name', type=str, default='train_RL_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--debug', type=int, nargs='*', default=0, help='number of samples')
    parser.add_argument('--steps_per_checkpoint', type=int, default=50000, help='Number of steps for checkpoint.')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoint', help='Checkpoint folder location')
    parser.add_argument('--log_folder', type=str, default='log', help='Log folder location')
    parser.add_argument('--log_file_name', type=str, default='train_log.txt', help='Log file name')
    parser.add_argument('--is_resume_from_checkpoint', type=int, default=1, help='Flag for resuming from last checkpoint')
    parser.add_argument('--logging_mode', type=str, default='a', help='logging mode')
    args = parser.parse_args()

    print('args.gpu: ', args.gpu)
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('args.device: ', args.device)

    args.dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    args.checkpoint_dir = '{}/{}'.format(args.dir, args.checkpoint_folder)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    args.log_dir = '{}/{}'.format(args.dir, args.log_folder)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/' + args.log_file_name, mode=args.logging_mode)
    logger.info(args)

    set_random_seed(args.seed)
    train(args)


if __name__ == '__main__':
    main()

