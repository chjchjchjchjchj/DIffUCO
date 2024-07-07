import pickle
import ipdb
from collections import Counter
import numpy as np
import time
from copy import deepcopy as dp
import pickle
from tqdm import trange

def is_valid_combination(combination, matrix):
    for i in range(len(combination)):
        for j in range(i + 1, len(combination)):
            if matrix[combination[i], combination[j]] > 0.5:
                return False
    return True

def find_combinations(matrix):
    n = matrix.shape[0]
    result = []

    def backtrack(start, current_combination):
        if is_valid_combination(current_combination, matrix):
            result.append(current_combination[:])
        
        for i in range(start, n):
            current_combination.append(i)
            backtrack(i + 1, current_combination)
            current_combination.pop()

    backtrack(0, [])
    results = [k for k in result if len(k)>1]
    return results

def check_overlap(matrix):
    overlapping_pairs = []
    num_balls = matrix.shape[0]
    
    for i in range(num_balls):
        for j in range(i + 1, num_balls):
            if matrix[i, j] > 0.5:
                overlapping_pairs.append((i, j))
    return overlapping_pairs


def cosine_matrix(vectors):
    dim = len(vectors[0])
    nums = len(vectors)
    Matrix = np.zeros((nums, nums))
    
    for i in range(nums):
        for j in range(nums):
            Matrix[i][j] = np.dot(vectors[i], vectors[j].T)

    return Matrix


def generate_unit_vectors(n, num_samples):
    """
    生成 num_samples 个 n 维的单位向量。

    参数:
    n (int): 向量的维度。
    num_samples (int): 生成的单位向量的数量。

    返回:
    numpy.ndarray: 形状为 (num_samples, n) 的单位向量数组。
    """
    # 生成随机向量
    seed = int(time.time())
    #seed = 1
    np.random.seed(seed)
    random_vectors = np.random.randn(num_samples, n)
    
    # 归一化向量以获得单位向量
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    unit_vectors = random_vectors / norms

    np.random.seed(int(time.time()))
    
    return unit_vectors

# path = "/home/chenhaojun/DIffUCO/draft/Data_for_solver.pkl"
# with open(path, 'rb') as f:
#     graphs = pickle.load(f)
# ipdb.set_trace()
# print(len(graphs)) # 2
# # graphs.keys() = dict_keys(['unit_vectors', 'overlap_id'])
# # len(graphs['overlap_id']) = 100 意思是100张图
# # graphs['overlap_id'][0] 第0张图的邻接表, list存了一堆元组
# # graphs['unit_vectors'][0].shape=(1000, 3)

def get_random_instance(dim, num_samples):
    # Data = {
    #     "unit_vectors":[],
    #     "overlap_id":[]
    # }
    Data = {}
    unit_vectors = generate_unit_vectors(dim, num_samples)
    matrix = cosine_matrix(unit_vectors)
    overlap_id = check_overlap(matrix)
    #print("unit_vectors=",unit_vectors)
    # print("cosin_matrix=",matrix)
    #print("overlap_id=",check_overlap(matrix))
    #print("indenpend set=",combinations)    
    
    Data["unit_vectors"] = unit_vectors
    Data["overlap_id"] = overlap_id

    return Counter(Data['overlap_id']), unit_vectors

    

