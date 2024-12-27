import streamlit as st
import pickle
import pandas as pd
import requests

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
from test_RL_agent import *
from utils import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#global logger
#logger = None

def recommended(args, logger):
    pred_labels, pred_labels_path, pred_labels_details, ndcg, recall, hit_ratio, precision, invalid_users = test(args, logger)
    return pred_labels, pred_labels_path, pred_labels_details

def main(args, logger):
    # Set page width and background color
    st.set_page_config(layout='wide', page_title='Product Recommendation System', page_icon='🎬', initial_sidebar_state='collapsed')
    # Title and sidebar
    st.title('Product Recommendation System')
    dataset = load_dataset(args.dataset)
    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')
    selected_user_key, selected_user_name = st.selectbox('Select a user:', [(selected_user_key, get_entity_details(dataset, "user", selected_user_key)) for selected_user_key in test_labels.keys()])
    #selected_user_name = get_entity_details(dataset, "user", selected_user_key)
    args.users = selected_user_key
    p_tab1, p_tab2 = st.tabs(["Historical Purchases", "Recommendations"])
    train_labels = {key: value for key, value in train_labels.items() if key == args.users}
    with p_tab1:
        st.subheader("Prior purchases of user \n User Key: " + str(selected_user_key) + "\n User Name: " + str(selected_user_name))
        p_col1, p_col2, p_col3, p_col4 = st.columns([1,1,1,4])
        p_col1.markdown("#### Product Key")
        p_col2.markdown("#### Product Name")
        p_col3.markdown("#### Brand")
        p_col4.markdown("#### Category")
        for prod in train_labels[args.users]:
            p_col1.write(prod)
            p_col2.write(get_entity_details(dataset, "product", prod))
            p_col3.write(str([get_entity_details(dataset, "brand", brand) for brand in dataset.produced_by.data[prod]]))
            p_col4.write(str([get_entity_details(dataset, "category", cat) for cat in dataset.belongs_to.data[prod]]))
    with p_tab2: 
        st.subheader("Product Recommendations")
        recommend_button = st.button('Recommend')
    
        # Main content
        if recommend_button:
            st.markdown("#### Recommended Products")
            args.MES_score_option = 0
            args.PAS_score_option = 0
            pred_labels, pred_labels_path, pred_labels_details = recommended(args, logger)
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Recommendations", "Recommendations with explainability", "Recommendations with MES", "Recommendations with MES and PPS"])
            if pred_labels:
                with tab1:
                    st.markdown("##### Recommended Products with basic recommendation features")
                    p_col1, p_col2, p_col3, p_col4 = st.columns([1,1,1,4])
                    p_col1.markdown("##### Product Key")
                    p_col2.markdown("##### Product Name")
                    p_col3.markdown("##### Brand")
                    p_col4.markdown("##### Category")
                    for prod in pred_labels[args.users]:
                        p_col1.write(prod)
                        p_col2.write(get_entity_details(dataset, "product", prod))
                        p_col3.write(str([get_entity_details(dataset, "brand", brand) for brand in dataset.produced_by.data[prod]]))
                        p_col4.write(str([get_entity_details(dataset, "category", cat) for cat in dataset.belongs_to.data[prod]]))
                with tab2: 
                    st.markdown("##### Recommended Products with explainability features")
                    col1, col2, col3 = st.columns([1,1,4])
                    col1.markdown("##### Product Key")
                    col2.markdown("##### Product Name")
                    col3.markdown("##### Explainability of the recommendation")
                    for user in pred_labels_path:
                        for prod_path in pred_labels_path[user]:
                            col1.write(prod_path[0])
                            col2.write(get_entity_details(dataset, "product", prod_path[0]))
                            col3.write(prod_path[1])
                with tab3:
                    st.markdown("##### Recommended Products powered by Max Explainability Score (MES) features")
                    args.MES_score_option = 1
                    args.PAS_score_option = 0
                    pred_labels, pred_labels_path, pred_labels_details = recommended(args, logger)
                    col1, col2, col3, col4 = st.columns([1,1,4,1])
                    col1.markdown("##### Product Key")
                    col2.markdown("##### Product Name")
                    col3.markdown("##### Explainability of the recommendation")
                    col4.markdown("##### MES")
                    for user in pred_labels_details:
                        for prod_path in pred_labels_details[user]:
                            col1.write(prod_path[0])
                            col2.write(get_entity_details(dataset, "product", prod_path[0]))
                            col3.write(prod_path[1])
                            col4.write(prod_path[2])
                with tab4:
                    st.subheader("Recommended Products powered by Max Explainability Score (MES) and Product Prioritization Score (PPS) features")
                    args.MES_score_option = 1
                    args.PAS_score_option = 1
                    pred_labels, pred_labels_path, pred_labels_details = recommended(args, logger)
                    col1, col2, col3, col4, col5 = st.columns([1,1,5,1,1])
                    col1.markdown("##### Product Key")
                    col2.markdown("##### Product Name")
                    col3.markdown("##### Explainability of the recommendation")
                    col4.markdown("##### MES")
                    col5.markdown("##### PPS")
                    for user in pred_labels_details:
                        for prod_path in pred_labels_details[user]:
                            col1.write(prod_path[0])
                            col2.write(get_entity_details(dataset, "product", prod_path[0]))
                            col3.write(prod_path[1])
                            col4.write(prod_path[2])
                            col5.write(prod_path[3])
            else:
                st.error("No recommendations found.")

if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=CELL, help='One of {cloth, beauty, cell, cd}')
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
    parser.add_argument('--log_file_name', type=str, default='test_agent_st_log', help='logging mode')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoint', help='Checkpoint folder location')
    parser.add_argument('--MES_score_option', type=int, default=0, help='Choose 0 for [Baseline], Choose 1 for [MES (Rewards Gain * Entropy Gain)], 2 for Only [Rewards Gain], 3 for Only [Entropy Gain], 4 for Only [Probs Gain], 5 for [Entopy Gain * Probs Gain], 6 for [Rewards Gain * Probs Gain], 7 for [Rewards Gain * Entopy Gain * Probs Gain], 8 for [Rewards Gain + Entopy Gain + Probs Gain]')
    parser.add_argument('--PAS_score_option', type=int, default=0, help='Choose 0 for [Baseline], Choose 1 for [PPS ()], 2 for Only [Score], 3 for Only [Prob], 4 for Only [Entropy], 5 for [Reward]')
    parser.add_argument('--run_number', type=int, default='1', help='logging mode')
    parser.add_argument('--is_only_run_specific_epoch', type=int, default=1, help='is_only_run_specific_epoch')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cpu')
    if args.gpu == '1':
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
    main(args, logger)