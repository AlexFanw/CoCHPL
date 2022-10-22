import json
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm import trange

if not os.path.exists('./final_data/'):
    os.mkdir('./final_data/')

review = pd.read_csv("/home/data/wwangbw/AlexCRS/review_filtered.csv", index_col=None)
print(review.head(3), '\n\n')
attribute = pd.read_csv("/home/data/wwangbw/AlexCRS/MAVE_filtered.csv", index_col=None)
print(attribute.head(3), '\n\n')
intention = pd.read_csv("/home/data/wwangbw/AlexCRS/folkscope_filtered.csv", index_col=None)
print(intention.head(3), '\n\n')

asin_index2id_dict = np.load("/home/data/wwangbw/AlexCRS/item_asin2id_dict.npy", allow_pickle=True).item()
reviewer_id_dict = np.load("/home/data/wwangbw/AlexCRS/reviewer_id2interaction_dict.npy", allow_pickle=True).item()
asin_id2index_dict = {v: k for k, v in asin_index2id_dict.items()}
reviewer_id2index_dict = {k: ind for ind, k in enumerate(list(reviewer_id_dict.keys()))}
reviewer_index2id_dict = {v: k for k, v in reviewer_id2index_dict.items()}
np.save('./final_data/item_index2id_dict.npy', asin_index2id_dict)
np.save('./final_data/item_id2index_dict.npy', asin_id2index_dict)
np.save('./final_data/user_id2index_dict.npy', reviewer_id2index_dict)
np.save('./final_data/user_index2id_dict.npy', reviewer_index2id_dict)

review = review.drop_duplicates(subset=['asin', 'reviewerID']).reset_index(drop=True)
review.to_csv('./final_data/all_user_review.csv', index=False)
review['index'] = review.index
review['asin_index'] = review['asin'].apply(lambda x: asin_id2index_dict[x])
review['reviewer_index'] = review['reviewerID'].apply(lambda x: reviewer_id2index_dict[x])
user_item_dict = {rid: None for rid in reviewer_index2id_dict.keys()}
for rid in tqdm(reviewer_index2id_dict.keys()):
    reviewer_review = review[review.reviewer_index == rid].sort_values(by='asin_index', ascending=True)
    user_item_dict[rid] = reviewer_review['asin_index'].tolist()
with open('./final_data/user_item.json', 'w') as fp:
    json.dump(user_item_dict, fp, sort_keys=True, indent=4)

trn_review = review.sample(frac=0.8, random_state=621).reset_index(drop=True)
dev_tst_review = review[~review['index'].isin(trn_review['index'])]
dev_review = dev_tst_review.sample(frac=0.5, random_state=621).reset_index(drop=True)
tst_review = dev_tst_review[~dev_tst_review['index'].isin(dev_review['index'])].reset_index(drop=True)
trn_review.drop(['index'], axis=1).to_csv('./final_data/trn_user_review.csv', index=False)
dev_review.drop(['index'], axis=1).to_csv('./final_data/dev_user_review.csv', index=False)
tst_review.drop(['index'], axis=1).to_csv('./final_data/tst_user_review.csv', index=False)

print(len(trn_review), len(dev_review), len(tst_review))

split_user_item_dict = {
    'train': {rid.item(): None for rid in trn_review['reviewer_index'].unique()},
    'valid': {rid.item(): None for rid in dev_review['reviewer_index'].unique()},
    'test': {rid.item(): None for rid in tst_review['reviewer_index'].unique()}
}
split_str = ['valid', 'test', 'train']
for ind, split in enumerate([dev_review, tst_review, trn_review]):
    for rid in tqdm(split['reviewer_index'].unique()):
        reviewer_review = split[split.reviewer_index == rid].sort_values(by='asin_index', ascending=True)
        split_user_item_dict[split_str[ind]][rid.item()] = reviewer_review['asin_index'].astype(int).tolist()
    with open('./final_data/review_dict_{}.json'.format(split_str[ind]), 'w') as fp:
        json.dump(split_user_item_dict[split_str[ind]], fp, sort_keys=True, indent=4)


def decide_nan(x):
    try:
        return math.isnan(x)
    except TypeError:
        return False


# list of features: intention, brand, first two categories, price, attribute (multiple keys)
item_feature_dict = {}
for i in trange(len(attribute)):
    asin_id = attribute.loc[i, 'asin']
    if asin_id not in item_feature_dict:
        item_feature_dict[asin_id] = {}
    if not decide_nan(attribute.loc[i, 'brand']):
        item_feature_dict[asin_id]['brand'] = attribute.loc[i, 'brand']
    if not decide_nan(attribute.loc[i, 'category']):
        first_two_categories = attribute.loc[i, 'category'].replace('[', '').replace(']', '').split("', '")[:2]
        item_feature_dict[asin_id]['category1'] = first_two_categories[0].replace("'", '')
        item_feature_dict[asin_id]['category2'] = first_two_categories[1].replace("'", '')
    if not decide_nan(attribute.loc[i, 'price']):
        item_feature_dict[asin_id]['price'] = attribute.loc[i, 'price']
    if not decide_nan(attribute.loc[i, 'attributes']):
        attribute_item_dict = json.loads(attribute.loc[i, 'attributes'])
        for k in attribute_item_dict:
            features = list(set(list(attribute_item_dict[k])))
            item_feature_dict[asin_id][k] = features

intention = intention[intention.score >= 0.99].reset_index(drop=True)
for i in trange(len(intention)):
    if intention.loc[i, 'item_a_id'] in item_feature_dict.keys():
        intention_parsed = intention.loc[i, 'assertion'].replace(
            'PersonX bought a product of Item A and a product of Item B because they both are', '').replace(
            'PersonX bought a product of Item A and a product of Item B because they both have', '').replace(
            'PersonX bought a product of Item A and a product of Item B because they are both', '').replace(
            'PersonX bought a product of Item A and a product of Item B because ', '').strip()
        if len(intention_parsed.split(' ')) > 10:
            continue
        if 'intention' not in item_feature_dict[intention.loc[i, 'item_a_id']].keys():
            item_feature_dict[intention.loc[i, 'item_a_id']]['intention'] = [intention_parsed]
        else:
            item_feature_dict[intention.loc[i, 'item_a_id']]['intention'].append(intention_parsed)
    if intention.loc[i, 'item_b_id'] in item_feature_dict.keys():
        intention_parsed = intention.loc[i, 'assertion'].replace(
            'PersonX bought a product of Item A and a product of Item B because they both are', '').replace(
            'PersonX bought a product of Item A and a product of Item B because they both have', '').replace(
            'PersonX bought a product of Item A and a product of Item B because they are both', '').replace(
            'PersonX bought a product of Item A and a product of Item B because ', '').strip()
        if len(intention_parsed.split(' ')) > 10:
            continue
        if 'intention' not in item_feature_dict[intention.loc[i, 'item_b_id']].keys():
            item_feature_dict[intention.loc[i, 'item_b_id']]['intention'] = [intention_parsed]
        else:
            item_feature_dict[intention.loc[i, 'item_b_id']]['intention'].append(intention_parsed)

for k in item_feature_dict.keys():
    if 'intention' in item_feature_dict[k].keys():
        item_feature_dict[k]['intention'] = list(set(item_feature_dict[k]['intention']))

with open('./final_data/item_dict_text.json', 'w') as fp:
    json.dump(item_feature_dict, fp, sort_keys=True, indent=4)

unique_feature_keys = {}
unique_feature_values = {}
for feature_dict in tqdm(item_feature_dict.values()):
    for k in feature_dict.keys():
        if k not in unique_feature_keys:
            unique_feature_keys[k] = None
        if type(feature_dict[k]) == list:
            for v in feature_dict[k]:
                if v not in unique_feature_values:
                    unique_feature_values[v] = None
        else:
            assert type(feature_dict[k]) in [str, float, int]
            if feature_dict[k] not in unique_feature_values:
                unique_feature_values[feature_dict[k]] = None

feature_key2id_dict = {k: ind for ind, k in enumerate(unique_feature_keys.keys())}
feature_id2key_dict = {v: k for k, v in feature_key2id_dict.items()}
feature_value2id_dict = {k: ind for ind, k in enumerate(unique_feature_values.keys())}
feature_id2value_dict = {v: k for k, v in feature_value2id_dict.items()}
np.save('./final_data/feature_key2id_dict.npy', feature_key2id_dict)
np.save('./final_data/feature_id2key_dict.npy', feature_id2key_dict)
np.save('./final_data/feature_value2id_dict.npy', feature_value2id_dict)
np.save('./final_data/feature_id2value_dict.npy', feature_id2value_dict)

item_dict = {asin_id2index_dict[asin_temp_id]: {
    'feature_index': [feature_key2id_dict[k] for k in item_feature_dict[asin_temp_id].keys()]} for asin_temp_id in
    tqdm(item_feature_dict.keys(), desc="Building item dict")}
with open('./final_data/item_dict.json', 'w') as fp:
    json.dump(item_dict, fp, sort_keys=True, indent=4)

item_dict_star = {asin_id2index_dict[asin_temp_id]: {'feature_index': []} for asin_temp_id in item_feature_dict.keys()}
for asinid in tqdm(item_feature_dict.keys(), desc="Building star item dict"):
    for k in item_feature_dict[asinid].keys():
        if type(item_feature_dict[asinid][k]) == list:
            for va in item_feature_dict[asinid][k]:
                item_dict_star[asin_id2index_dict[asinid]]['feature_index'].append(feature_value2id_dict[va])
        else:
            item_dict_star[asin_id2index_dict[asinid]]['feature_index'].append(
                feature_value2id_dict[item_feature_dict[asinid][k]])
with open('./final_data/item_dict.json', 'w') as fp:
    json.dump(item_dict_star, fp, sort_keys=True, indent=4)
