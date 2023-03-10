from utils import *

# Get the list of users for training the model
#uids = list(env.kg(USER).keys())
uids = [22341, 22342]

# Get corresponding train labels based on the users in the train dataset
train_labels = load_labels('beauty', 'train')
train_labels = dict(sorted((key, train_labels[key]) for key in train_labels.keys() if key in uids))
print('train_labels: ', train_labels)

kg = load_kg('beauty')
relations_nodes = kg('user', 22341)
print(relations_nodes)
relations_nodes = kg('user', 22342)
print(relations_nodes)