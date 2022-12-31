import json

folkscope_item = json.load(open('../../datasets/raw_data/folkscope_star_v1/Graph_generate_data/item_dict.json', 'r'))
print(len(folkscope_item))
folkscope_user = json.load(open('../../datasets/raw_data/folkscope_star_v1/Graph_generate_data/user_item.json', 'r'))
print(len(folkscope_user))
entity_total_num = len(folkscope_item) + len(folkscope_user)

raise Exception
entity2id_folkscope = open('../../datasets/transE_data/folkscope_v1/entity2id.txt', 'w')
entity2id_folkscope.write("{}\n".format(entity_total_num))
entity_dict = {}
for i in range(len(folkscope_user)):
    entity2id_folkscope.write("user_{}\t{}\n".format(i, i))
    entity_dict["user_{}".format(i)] = i
for i in range(len(folkscope_item)):
    entity2id_folkscope.write("item_{}\t{}\n".format(i, len(folkscope_user) + i))
    entity_dict["item_{}".format(i)] = len(folkscope_user) + i
entity2id_folkscope.close()

print(entity_dict['item_33460'])
train_edge = json.load(open('../../datasets/raw_data/folkscope/UI_Interaction_data/review_dict_train.json', 'r'))
dev_edge = json.load(open('../../datasets/raw_data/folkscope/UI_Interaction_data/review_dict_valid.json', 'r'))
tst_edge = json.load(open('../../datasets/raw_data/folkscope/UI_Interaction_data/review_dict_test.json', 'r'))
train_edge_num = sum([len(train_edge[u]) for u in train_edge.keys()])
dev_edge_num = sum([len(dev_edge[u]) for u in dev_edge.keys()])
tst_edge_num = sum([len(tst_edge[u]) for u in tst_edge.keys()])
print(train_edge_num, dev_edge_num, tst_edge_num)
train2id = open('../../datasets/transE_data/folkscope_v1/train2id.txt', 'w')
train2id.write("{}\n".format(train_edge_num))
for u in train_edge.keys():
    for i in train_edge[u]:
        train2id.write(
            "{}\t{}\t{}\n".format(entity_dict['user_{}'.format(u)], entity_dict['item_{}'.format(i)], 0))
train2id.close()

valid2id = open('../../datasets/transE_data/folkscope_v1/valid2id.txt', 'w')
valid2id.write("{}\n".format(dev_edge_num))
for u in dev_edge.keys():
    for i in dev_edge[u]:
        valid2id.write(
            "{}\t{}\t{}\n".format(entity_dict['user_{}'.format(u)], entity_dict['item_{}'.format(i)], 0))
valid2id.close()

test2id = open('../../datasets/transE_data/folkscope_v1/test2id.txt', 'w')
test2id.write("{}\n".format(tst_edge_num))
for u in tst_edge.keys():
    for i in tst_edge[u]:
        test2id.write(
            "{}\t{}\t{}\n".format(entity_dict['user_{}'.format(u)], entity_dict['item_{}'.format(i)], 0))
test2id.close()
