import pickle
import ipdb
from collections import Counter

# path = "/home/chenhaojun/DIffUCO/draft/Data_for_solver.pkl"
# with open(path, 'rb') as f:
#     graphs = pickle.load(f)
# ipdb.set_trace()
# print(len(graphs)) # 2
# # graphs.keys() = dict_keys(['unit_vectors', 'overlap_id'])
# # len(graphs['overlap_id']) = 100 意思是100张图
# # graphs['overlap_id'][0] 第0张图的邻接表, list存了一堆元组
# # graphs['unit_vectors'][0].shape=(1000, 3)

def get_random_instance(path, idx):
    with open(path, 'rb') as f:
        graphs = pickle.load(f)

    return Counter(graphs['overlap_id'][idx])

    

