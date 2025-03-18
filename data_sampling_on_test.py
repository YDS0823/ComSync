#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import random
import json
import pickle
from tqdm import tqdm
import jsonlines

def data_sampling(dataset_path,sampled_id_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        len=sum(1 for _ in f)
    index_lst = list(np.arange(0, len, 1))
    random.shuffle(index_lst)
    index_lst = index_lst[0:252]
    index_lst.sort()
    with open(sampled_id_path, "wb") as fw:
        pickle.dump(index_lst, fw)
    sampled_lst = []
    with open(dataset_path, encoding="utf-8") as fr:
        for id, line in enumerate(fr.readlines()):
            if id in index_lst:
                sampled_lst.append(json.loads(line))
    with jsonlines.open(output_path, "w") as fw:
        fw.write_all(sampled_lst)

if __name__ == "__main__":
    dataset_path="test.jsonl"
    sampled_id_path="test_sampled_id.pkl"
    output_path="test_sampled.jsonl"
    data_sampling(dataset_path,sampled_id_path,output_path)
