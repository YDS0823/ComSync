from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import pickle
import pandas as pd
from random import sample
import random
import pickle
import argparse


def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def generate_sample_pairs(num_pairs, sample_size, range_start, range_end):

    sample_pairs = []
    for _ in range(num_pairs):
        
        sample = random.sample(range(range_start, range_end + 1), sample_size)
        sample_pairs.append(sample)
    return sample_pairs

def expert_retrieval_sampled(test_sample_path,train_path,test_path,output_path):
    with open(test_sample_path, "rb") as fr:
        sampled_ids = pickle.load(fr)
    featureList = ['NMS', 'NMT', 'NML', 'NMC', 'NNSPR', 'NNTPR', 'NTOD', 'NSOD', 'TS_0', 'TS_1', 'TS_2']
    train_expert = pd.read_csv(train_path)
    train_expert_features = train_expert.loc[:, featureList]
    train_expert_features = train_expert_features.values.tolist()
    train_len = len(train_expert_features)

    test_expert = pd.read_csv(test_path)
    test_expert_features = test_expert.loc[:, featureList]
    test_expert_features = test_expert_features.values.tolist()
    results_ids = []
    for id in tqdm(sampled_ids):
        vfeatures = test_expert_features[id]
        cosine_lst = []
        for tfeatures in train_expert_features:
            cosine_lst.append(np.dot(vfeatures, tfeatures) / (norm(vfeatures) * norm(tfeatures)))
       
        sorted_id = sorted(range(train_len), key=lambda k: cosine_lst[k], reverse=True)[:100]
        results_ids.append(sorted_id)
    with open(output_path, "wb") as fw:
        pickle.dump(results_ids, fw)


def dense_retrieval_sampled(test_sample_path,train_path,test_path,output_path):
    with open(train_path, "rb") as fr:
        train = pickle.load(fr)
    with open(test_path, "rb") as fr:
        test = pickle.load(fr)
    train_len = len(train)
    # query
    results_ids = []
    for embed_q in tqdm(test):
        cosine_lst = []
        for embed_v in train:
            cosine_lst.append(np.dot(embed_q, embed_v) / (10))
        sorted_id = sorted(range(train_len), key=lambda k: cosine_lst[k], reverse=True)[:100]
        results_ids.append(sorted_id)
    # print(results_ids[1])
    with open(test_sample_path, "rb") as fr:
        sampled_ids = pickle.load(fr)
    sampled_dense_ids = []
    for id in sampled_ids:
        sampled_dense_ids.append(results_ids[id])
    with open(output_path, "wb") as fw:
        pickle.dump(sampled_dense_ids, fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--dataset", default="", type=str, required=True,
                        help="choose a dataset")
    args = parser.parse_args()
    dataset=args.dataset
    print(dataset)
    length=len(read_pkl_file(f"./dataset/{dataset}/test_sampled_id.pkl"))
    range_end=len(read_pkl_file(f"./dataset/{dataset}/Codebert/train_dense_DB.pkl"))
    sample_pairs = generate_sample_pairs(num_pairs=length,sample_size=10,range_start=1,range_end=range_end)
    with open(f'ComSync/retrieval/{dataset}/random_id_sampled.pkl', 'wb') as f:
        pickle.dump(sample_pairs, f)

    test_sample_path=f"ComSync/dataset/{dataset}/test_sampled_id.pkl"
    train_path=f"ComSync/dataset/{dataset}/Expert/featuresForTrain.csv"
    test_path=f"ComSync/dataset/{dataset}/Expert/featuresForTest.csv"
    output_path=f"ComSync/retrieval/{dataset}/expert_id_sampled.pkl"
    expert_retrieval_sampled(test_sample_path,train_path,test_path,output_path)

    train_path=f"ComSync/dataset/{dataset}/Codebert/train_dense_DB.pkl"
    test_path=f"ComSync/dataset/{dataset}/Codebert/test_dense_DB.pkl"
    output_path=f"ComSync/retrieval/{dataset}/dense_id_sampled.pkl"
    dense_retrieval_sampled(test_sample_path,train_path,test_path,output_path)

    

