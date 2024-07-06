
import kg_env
#import knowledge_graph
from utils import *
from knowledge_graph import *
from kg_env import *


# Get the list of users for training the model
#uids = list(env.kg(USER).keys())
#batch_uids = [22341, 22342]
batch_uids = {22012: [4266]}
#batch_uids = {22341: [10463, 4215, 8177, 591, 12007, 10105, 6316, 331, 1317, 10430, 5635, 6767, 587, 8322, 5395, 6730, 10244, 5213, 7817, 5884, 6451, 8365, 359, 9389, 6629, 10732, 8185, 599, 5334, 11798, 9004, 9116, 3135], 22342: [1885, 8885, 5210, 1768, 2163, 8417, 1153, 5672, 5063, 257, 1442, 1847, 2500, 6131]}

#batch_uids = {22341: [3253, 9439, 11393, 5432, 8616, 3412, 10474, 540, 2823, 8720, 8889, 1812, 9568], 22342: [1117, 9042, 6758, 711, 2478]}
#batch_curr_further_processing = {uid: 1 for uid in batch_uids}


# Get corresponding train labels based on the users in the train dataset
train_labels = load_labels('beauty', 'train')
#train_labels = dict(sorted((key, train_labels[key][0]) for key in train_labels.keys() if key in uids))
#train_labels = dict(sorted((key, train_labels[key]) for key in train_labels.keys() if key in uids))
#print('train_labels: ', train_labels)
#batch_path = {22341: [(SELF_LOOP, USER, 22341)], 22342: [(SELF_LOOP, USER, 22342)]}
#batch_path = {22341: [(SELF_LOOP, USER, 22341), ('mentions', WORD, 2118)], 22342: [(SELF_LOOP, USER, 22342), ('mentions', WORD, 8874)]}
#batch_path = {22341: [(SELF_LOOP, USER, 22341), ('mentions', WORD, 2118), ('mentions', USER, 14929)], 22342: [(SELF_LOOP, USER, 22342), ('mentions', WORD, 8874), ('described_as', PRODUCT, 1885)]}
#batch_path = [[('self_loop', 'user', 22341)], [('self_loop', 'user', 22342)]]
batch_path = [[('self_loop', 'user', 22341)], [('self_loop', 'user', 22344)]]
#path = [path for path in batch_path if path[0][2] == 22341][0]
#print(batch_path)
#print(path)

kg = load_kg('beauty')
#relations_nodes = kg('user', 22341)
#print(relations_nodes)
#relations_nodes = kg('user', 22342)
#print(relations_nodes)
dataset = load_dataset('beauty', mode='test')
#print(get_entity_details(dataset, 'user', 22012))
#print(get_entity_details(dataset, 'user', 22013))
#user_product_hop_actions = batch_get_user_product_path_actions_actual(dataset_str='beauty', kg=kg, batch_uids_targetlabel=batch_uids, batch_path=batch_path, batch_curr_further_processing=batch_curr_further_processing, is_train=0, is_debug=0)
#user_product_hop_actions = knowledge_graph.batch_get_user_products_actions_actual('beauty', kg, batch_uids, batch_path, batch_curr_further_processing, is_train=1, is_debug=0)

#print('user_product_hop_actions: ', user_product_hop_actions)
#print([user_product_hop_actions[key] for key in user_product_hop_actions if key == 22341][0])

#check_user_product_path('beauty', kg, batch_uids=batch_uids, mode='test')
print('execution_timestamp={}'.format(datetime.now()))

'''
path_pattern = ('self_loop', 'mentions')
print(path_pattern)
pattern = ('self_loop', 'mentions', 'mentions', 'purchase')


if any((path_pattern == pattern[i:i+len(path_pattern)]) for i in range(len(pattern)-len(path_pattern)+1)):
    print('yes')
else:
    print('no')

print(tuple(path_pattern))
print(tuple(pattern))


big_dict = [('apple', 'red'), ('banana', 'yellow'), ('orange', 'orange'), ('pear', 'green')]
match_dict = [('apple', 'red'), ('pear', 'green')]

# Get index value of matching keys in big_dict
#matching_indexes = [(i, key) for i, key in enumerate(big_dict) if key in match_dict]
matching_indexes = [(big_dict.index(item), item) for item in match_dict]

print(matching_indexes)

'''
'''

def batch_get_idx_curr_actions_actual(batch_curr_actions, batch_actions_actual, is_debug=0):
    # Get the batch actions indexes followed by the agent to reach the target label
    idx_batch_curr_actions_actual = [
        [batch_curr_actions[key].index(item) for item in value if item in batch_actions_actual[key]] for key, value in
        batch_curr_actions.items()]
    if is_debug == 1:
        print('idx_batch_curr_actions_actual: ', idx_batch_curr_actions_actual)
    return idx_batch_curr_actions_actual

batch_path = [[('self_loop', 'user', 1263)], [('self_loop', 'user', 1814)], [('self_loop', 'user', 2190), ('purchase', 'product', 7791)], [('self_loop', 'user', 2303)], [('self_loop', 'user', 2491)], [('self_loop', 'user', 3761), ('mentions', 'word', 13230)], [('self_loop', 'user', 4104)], [('self_loop', 'user', 5095)], [('self_loop', 'user', 5902)], [('self_loop', 'user', 6365)], [('self_loop', 'user', 6923)], [('self_loop', 'user', 9187)], [('self_loop', 'user', 10105)], [('self_loop', 'user', 10385)], [('self_loop', 'user', 10542)], [('self_loop', 'user', 11360)], [('self_loop', 'user', 11435), ('mentions', 'word', 17368)], [('self_loop', 'user', 12695)], [('self_loop', 'user', 13004)], [('self_loop', 'user', 13340), ('mentions', 'word', 7316)], [('self_loop', 'user', 13992), ('mentions', 'word', 3306)], [('self_loop', 'user', 14955), ('purchase', 'product', 3161), ('also_viewed', 'related_product', 4834)], [('self_loop', 'user', 16658)], [('self_loop', 'user', 17009)], [('self_loop', 'user', 17266)], [('self_loop', 'user', 17810)], [('self_loop', 'user', 17844), ('mentions', 'word', 2396)], [('self_loop', 'user', 20855), ('purchase', 'product', 9187)], [('self_loop', 'user', 21217), ('mentions', 'word', 12521)], [('self_loop', 'user', 21633)], [('self_loop', 'user', 21793)], [('self_loop', 'user', 21850)]]
uids = {1263: 0, 1814: 0, 2190: 516, 2303: 0, 2491: 0, 3761: 5179, 4104: 0, 5095: 3468, 5902: 0, 6365: 0, 6923: 0, 9187: 0, 10105: 0, 10385: 4550, 10542: 0, 11360: 11457, 11435: 6517, 12695: 0, 13004: 0, 13340: 4246, 13992: 11946, 14955: 7745, 16658: 0, 17009: 0, 17266: 0, 17810: 0, 17844: 5506, 20855: 8013, 21217: 7716, 21633: 0, 21793: 0, 21850: 0}
batch_act_idx = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
batch_act_actual_idx = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [249, 250], [], [], [], [], [], [], [], [], []]
batch_curr_further_processing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
kg = load_kg('beauty')

batch_curr_actions = [[('self_loop', 1263)], [('self_loop', 1814)], [('self_loop', 7791)], [('self_loop', 2303)], [('self_loop', 2491)], [('self_loop', 13230)], [('self_loop', 4104)], [('self_loop', 5095)], [('self_loop', 5902)], [('self_loop', 6365)], [('self_loop', 6923)], [('self_loop', 9187)], [('self_loop', 10105)], [('self_loop', 10385)], [('self_loop', 10542)], [('self_loop', 11360)], [('self_loop', 17368)], [('self_loop', 12695)], [('self_loop', 13004)], [('self_loop', 7316)], [('self_loop', 3306)], [('self_loop', 4834), ('also_viewed', 4158), ('also_bought', 10093), ('also_viewed', 10390), ('also_bought', 2194), ('also_bought', 11418), ('also_viewed', 9978), ('also_viewed', 10819), ('also_viewed', 4668), ('also_bought', 4877), ('also_viewed', 1724), ('also_viewed', 3080), ('also_bought', 3080), ('also_viewed', 6389), ('also_viewed', 11034), ('also_bought', 1543), ('also_bought', 707), ('also_viewed', 7020), ('also_viewed', 9454), ('also_viewed', 2700), ('also_viewed', 11676), ('also_viewed', 8958), ('also_viewed', 10930), ('also_viewed', 2660), ('also_viewed', 10464), ('also_viewed', 11797), ('also_viewed', 6193), ('also_viewed', 1674), ('also_viewed', 10669), ('also_bought', 7178), ('also_viewed', 7484), ('also_bought', 11714), ('also_viewed', 3990), ('also_viewed', 5508), ('also_bought', 1716), ('also_viewed', 10848), ('also_viewed', 10391), ('also_bought', 1720), ('also_viewed', 9408), ('also_viewed', 11672), ('also_viewed', 11738), ('also_viewed', 206), ('also_viewed', 1666), ('also_viewed', 6669), ('also_bought', 8165), ('also_viewed', 4364), ('also_viewed', 3131), ('also_viewed', 4402), ('also_viewed', 8454), ('also_viewed', 7293), ('also_bought', 10036), ('also_viewed', 9898), ('also_viewed', 1387), ('also_viewed', 986), ('also_bought', 1146), ('also_viewed', 6510), ('also_bought', 6510), ('also_bought', 4911), ('also_viewed', 8747), ('also_viewed', 6565), ('also_viewed', 4640), ('also_viewed', 7882), ('also_viewed', 4256), ('also_viewed', 11442), ('also_viewed', 5518), ('also_viewed', 12069), ('also_viewed', 3063), ('also_bought', 11245), ('also_bought', 3063), ('also_viewed', 11245), ('also_bought', 7641), ('also_viewed', 1148), ('also_bought', 7397), ('also_bought', 4965), ('also_viewed', 8561), ('also_viewed', 207), ('also_viewed', 7619), ('also_viewed', 9670), ('also_viewed', 6036), ('also_bought', 10866), ('also_bought', 356), ('also_viewed', 10767), ('also_viewed', 11269), ('also_viewed', 9873), ('also_viewed', 8358), ('also_bought', 8358), ('also_viewed', 10956), ('also_bought', 9200), ('also_viewed', 3948), ('also_viewed', 2636), ('also_viewed', 3144), ('also_viewed', 2152), ('also_viewed', 1795), ('also_viewed', 850), ('also_bought', 9015), ('also_viewed', 9015), ('also_viewed', 7881), ('also_viewed', 4624), ('also_bought', 11197), ('also_bought', 610), ('also_viewed', 11291), ('also_bought', 1888), ('also_bought', 8861), ('also_bought', 4478), ('also_viewed', 5215), ('also_viewed', 11700), ('also_bought', 11694), ('also_bought', 749), ('also_bought', 1707), ('also_viewed', 2103), ('also_bought', 2103), ('also_bought', 10971), ('also_viewed', 2995), ('also_bought', 4403), ('also_viewed', 4403), ('also_viewed', 4444), ('also_viewed', 1491), ('also_bought', 4597), ('also_viewed', 4878), ('also_viewed', 7198), ('also_viewed', 10172), ('also_bought', 5781), ('also_bought', 4004), ('also_viewed', 4004), ('also_bought', 7163), ('also_bought', 3153), ('also_bought', 6567), ('also_bought', 6962), ('also_viewed', 6962), ('also_bought', 6371), ('also_bought', 546), ('also_bought', 5042), ('also_viewed', 1130), ('also_bought', 1130), ('also_viewed', 6599), ('also_viewed', 3815), ('also_viewed', 6786), ('also_bought', 6650), ('also_viewed', 10597), ('also_bought', 11009), ('also_bought', 4596), ('also_bought', 8901), ('also_viewed', 965), ('also_bought', 7673), ('also_bought', 11445), ('also_viewed', 621), ('also_bought', 1412), ('also_viewed', 10213), ('also_bought', 1054), ('also_bought', 1826), ('also_viewed', 1826), ('also_viewed', 8833), ('also_bought', 4385), ('also_bought', 6120), ('also_viewed', 5281), ('also_bought', 5281), ('also_viewed', 7012), ('also_viewed', 5914), ('also_bought', 5914), ('also_viewed', 1992), ('also_bought', 997), ('also_viewed', 9851), ('also_bought', 9851), ('also_viewed', 10351), ('bought_together', 10351), ('also_bought', 10351), ('also_viewed', 7497), ('also_bought', 7497), ('also_viewed', 15), ('also_viewed', 11129), ('also_viewed', 3382), ('also_viewed', 533), ('bought_together', 4188), ('also_viewed', 4188), ('also_bought', 4188), ('also_bought', 10209), ('also_viewed', 4757), ('also_bought', 11608), ('also_viewed', 9268), ('also_bought', 2618), ('also_bought', 10764), ('also_viewed', 3223), ('also_viewed', 9880), ('also_bought', 4519), ('also_bought', 10369), ('also_bought', 10370), ('also_bought', 1477), ('also_viewed', 3602), ('also_bought', 3602), ('also_viewed', 5827), ('also_bought', 5827), ('also_bought', 4759), ('also_viewed', 4759), ('also_viewed', 9621), ('also_bought', 1069), ('also_bought', 6387), ('also_bought', 7125), ('also_bought', 7081), ('also_bought', 11770), ('also_bought', 7338), ('also_viewed', 7279), ('also_bought', 7279), ('also_bought', 5286), ('also_bought', 2725), ('also_viewed', 2725), ('also_bought', 4094), ('also_bought', 3106), ('also_bought', 4249), ('also_viewed', 4249), ('also_bought', 10328), ('also_bought', 5980), ('also_bought', 7377), ('also_viewed', 7377), ('also_bought', 3104), ('also_viewed', 10946), ('also_bought', 10946), ('also_bought', 4006), ('also_viewed', 3561), ('also_bought', 3561), ('also_bought', 3758), ('also_bought', 9709), ('also_bought', 2598), ('also_bought', 8179), ('also_viewed', 8179), ('also_bought', 7836), ('also_bought', 1895), ('also_viewed', 10283), ('also_bought', 10283), ('also_bought', 2546), ('also_bought', 6864), ('also_bought', 5429), ('also_bought', 11453), ('also_viewed', 1999), ('also_bought', 1999), ('also_viewed', 9794), ('also_bought', 9794), ('also_bought', 2136), ('also_viewed', 2136), ('also_viewed', 5678), ('also_bought', 5678), ('also_bought', 6141), ('also_bought', 535), ('also_viewed', 535), ('also_bought', 3331), ('also_viewed', 3331), ('also_bought', 9260), ('also_viewed', 9260), ('also_bought', 6529), ('also_viewed', 6529), ('also_viewed', 7745), ('also_bought', 7745)], [('self_loop', 16658)], [('self_loop', 17009)], [('self_loop', 17266)], [('self_loop', 17810)], [('self_loop', 2396)], [('self_loop', 9187)], [('self_loop', 12521)], [('self_loop', 21633)], [('self_loop', 21793)], [('self_loop', 21850)]]
batch_action = {batch_path[batch_curr_actions.index(action)][0][2]: action for action in batch_curr_actions}
actual_actions = knowledge_graph.batch_get_user_product_path_actions_actual('beauty', kg, uids, batch_path, 'train', is_debug=0)
print('actual_actions: ', actual_actions)

# print('self._batch_curr_actions: ', self._batch_curr_actions)
print('batch_action: ', batch_action)
#print('self._batch_curr_actions_actual: ', self._batch_curr_actions_actual)



batch_curr_actions_actual_idx = batch_get_idx_curr_actions_actual(batch_curr_actions = batch_action, batch_actions_actual = actual_actions, is_debug=0)
print('batch_curr_actions_actual_idx: ', batch_curr_actions_actual_idx)

#for path in batch_path:
#    print('path, batch_act_idx[batch_path.index(path)], batch_act_actual_idx[batch_path.index(path)], uids[path[0][2]]: ',path, batch_act_idx[batch_path.index(path)], batch_act_actual_idx[batch_path.index(path)], uids[path[0][2]])
'''
'''
from kg_env import BatchKGEnvironment

env = BatchKGEnvironment('beauty', 250, max_path_len=3, state_history=1)

new_path = [('self_loop', 'user', 22341)]
reward = env._get_reward(new_path, 0, is_train=0, is_debug=0)
# print('reward: ', reward)
'''

import glob
import os.path
folder_path = './tmp/Amazon_Clothing/train_transe_model/checkpoint/'
file_type = r'/*.ckpt'
files = glob.glob(folder_path + file_type)
print('files, len(files): ', files, len(files))
'''if len(files) > 2:
    max_file = max(files, key=os.path.getctime)
else:
    max_file = None
print'''