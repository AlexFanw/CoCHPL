import pickle
from easydict import EasyDict
# import os
# os.chdir("../")
class MovieDataset(object):
    def __init__(self):
        with open('./datasets/processed_data/movie/kg.pkl', 'rb') as f:
            kg=pickle.load(f)
        entity_id=list(kg.G['user'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'user',m)
        
        entity_id=list(kg.G['item'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'item',m)
        
        entity_id=list(kg.G['feature'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'feature',m)

