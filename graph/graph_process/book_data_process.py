import pickle
from easydict import EasyDict
# import os
# os.chdir("../")

class BookDataset(object):
    def __init__(self):
        with open('./datasets/processed_data/book/kg.pkl','rb') as f:
            kg=pickle.load(f)
        entity_id=list(kg.G['user'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'user',m)
        
        entity_id=list(kg.G['item'].keys())
        m=EasyDict(id=entity_id, value_len=max(entity_id)+1)
        setattr(self,'item',m)
        
        entity_id=list(kg.G['feature'].keys())
        m=EasyDict(id=entity_id, value_len=max(max(entity_id)+1,988))
        setattr(self,'feature',m)

