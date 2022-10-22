import math

import pandas as pd

electronic_review = pd.read_csv("/home/data/wwangbw/Electronics.csv", index_col=None,
                                names=['asin', 'reviewerID', 'rating', 'time'], header=None).drop(['time'], axis=1)
clothing_review = pd.read_csv("/home/data/wwangbw/Clothing_Shoes_and_Jewelry.csv", index_col=None,
                              names=['asin', 'reviewerID', 'rating', 'time'], header=None).drop(['time'], axis=1)

print(electronic_review.head(10))
print(clothing_review.head(10))
print(len(electronic_review), len(clothing_review))

MAVE_data = pd.read_csv("/home/data/wwangbw/AmazonMetadata_merge_MAVE.csv", index_col=None)

print(MAVE_data.head(3))
print(len(MAVE_data.asin.unique()))


def decide_nan(x):
    try:
        return math.isnan(x)
    except TypeError:
        return False


MAVE_data['attribute_empty'] = MAVE_data['attributes'].apply(lambda x: decide_nan(x))
MAVE_data_attr = MAVE_data[~MAVE_data.attribute_empty]
print(len(MAVE_data_attr), len(MAVE_data_attr.asin.unique()))

electronic_folkscope = pd.read_csv("/home/data/wwangbw/data_FINAL/elec_FINAL.csv", index_col=None)
clothing_folkscope = pd.read_csv("/home/data/wwangbw/data_FINAL/clothing_FINAL.csv", index_col=None)

electronic_folkscope = electronic_folkscope[electronic_folkscope.score >= 0.9].reset_index(drop=True)
clothing_folkscope = clothing_folkscope[clothing_folkscope.score >= 0.9].reset_index(drop=True)

electronic_folkscope_id_list = list(
    set(electronic_folkscope['item_a_id'].tolist() + electronic_folkscope['item_b_id'].tolist()))
clothing_folkscope_id_list = list(
    set(clothing_folkscope['item_a_id'].tolist() + clothing_folkscope['item_b_id'].tolist()))
total_id_list = electronic_folkscope_id_list + clothing_folkscope_id_list
print(len(electronic_folkscope_id_list), len(clothing_folkscope_id_list))

MAVE_data_attr_folkscope = MAVE_data_attr[
    MAVE_data_attr.asin.isin(electronic_folkscope_id_list + clothing_folkscope_id_list)].reset_index(drop=True)
print(len(MAVE_data_attr_folkscope), len(MAVE_data_attr_folkscope.asin.unique()))

asin_list = list(MAVE_data_attr_folkscope.asin.unique())
clothing_review_attr_folkscope = clothing_review[clothing_review.asin.isin(asin_list)].reset_index(
    drop=True)
electronic_review_attr_folkscope = electronic_review[
    electronic_review.asin.isin(asin_list)].reset_index(
    drop=True)
print(len(clothing_review_attr_folkscope) + len(electronic_review_attr_folkscope))
total_review_record = pd.concat([clothing_review_attr_folkscope, electronic_review_attr_folkscope], ignore_index=True)

from collections import Counter

review_user_record = Counter(total_review_record['reviewerID'])
review_user_record_sorted = {k: v for k, v in sorted(review_user_record.items(), key=lambda item: item[1])}

Counter(review_user_record_sorted.values())
print(sum(review_user_record_sorted.values()))
reviewer_id_dict = {k: v for k, v in review_user_record_sorted.items() if v > 1}

import numpy as np

np.save('/home/data/wwangbw/AlexCRS/reviewer_id2interaction_dict.npy', reviewer_id_dict)
np.save('/home/data/wwangbw/AlexCRS/item_asin2id_dict.npy', {ind: i for ind, i in enumerate(asin_list)})

review_record = total_review_record[total_review_record.reviewerID.isin(reviewer_id_dict.keys())].reset_index(drop=True)
print(len(review_record))
review_record.to_csv("/home/data/wwangbw/AlexCRS/review_filtered.csv", index=False)

folkscope_total = pd.concat([clothing_folkscope, electronic_folkscope], ignore_index=True)
folkscope_total = folkscope_total[folkscope_total.score >= 0.9]
folkscope_selected = folkscope_total[
    (folkscope_total.item_a_id.isin(asin_list)) | (folkscope_total.item_b_id.isin(asin_list))].reset_index(drop=True)
print(len(folkscope_selected))
folkscope_selected.to_csv("/home/data/wwangbw/AlexCRS/folkscope_filtered.csv", index=False)

MAVE_data_filtered = MAVE_data[MAVE_data.asin.isin(asin_list)].reset_index(drop=True)
print(len(MAVE_data_filtered))
MAVE_data_filtered.to_csv("/home/data/wwangbw/AlexCRS/MAVE_filtered.csv", index=False)
