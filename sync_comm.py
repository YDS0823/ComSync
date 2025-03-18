import openai
import pickle
import json
from utils.concat_rets import process_line
import random
import numpy as np
import jsonlines
from tqdm import tqdm
# from tenacity import (
#     retry,
#     retry_if_exception_type,
#     wait_random_exponential,
# )  # for exponential backoff


# @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIError)))
def collect_one_gpt(prompt, api_key,temperature):
    openai.api_key = api_key
    ret = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=prompt,
        max_tokens=50,
        temperature=temperature,
        n=10,
        top_p=0.95
    )
    samples = ret['choices']
    candidates = []
    for id, i in enumerate(samples):
        candi_lst = i['message']['content'].strip().split("\n")
        candi = ""
        for snippet in candi_lst:
            if snippet.endswith("END_OF_DEMO"):
                snippet = snippet.split("END_OF_DEMO")[0]
                candi += snippet.strip() + " END_OF_DEMO"
                break
            else:
                candi += snippet.strip() + " "
        candi = candi.strip()
        if candi != "" and len(candi.split()) > 1 and candi not in candidates:
            candidates.append(candi)
    return candidates


from openai import OpenAI
def collect_one(prompt, api_key,temperature,basemodel):
    
    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = api_key
    )
    #llama3-70b
    ret = client.chat.completions.create(
        model=f"meta/{basemodel}-instruct",
        messages=prompt,
        temperature=temperature,
        top_p=0.95,
        max_tokens=50,
        stream=False
    )
    candi_lst =ret.choices[0].message.content

    candidates = []
    candi = ""
    if candi_lst.endswith("END_OF_DEMO"):
        candi_lst = candi_lst.split("END_OF_DEMO")[0]
        candi += candi_lst.strip() + " END_OF_DEMO"
    else:
        candi += candi_lst.strip() + " "
    candi = candi.strip()
    if candi != "" and len(candi.split()) > 1 and candi not in candidates:
        candidates.append(candi)
    return candidates



def collect_all_retrieval(dataset,model,train_path,rtr_path, test_path, out_path, shots, api_key, start,temperature):
    train = []
    with open(train_path, encoding="utf-8") as fr:
        for line in fr.readlines():
            train.append(json.loads(line))

    with open(rtr_path, "rb") as fr:
        rtr_ids = pickle.load(fr)

    with open(test_path, encoding="utf-8") as fr:
        count = 0
        for id, (line, example_ids) in tqdm(enumerate(zip(fr.readlines(), rtr_ids)), total=len(rtr_ids)):
            count += 1
            # todo: START FROM WHERE
            if count <= start:
                continue
            src_method = []
            dst_method = []
            src_doc = []
            dst_doc = []
            for id in example_ids[:shots]:
                example = train[id]
                if dataset=="Hebcup":
                    src_method.append(example['src_method'])
                    dst_method.append(example['dst_method'])
                    src_doc.append(example['src_desc'])
                    dst_doc.append(example['dst_desc'])
                else:
                    src_method.append(example['old_code'])
                    dst_method.append(example['new_code'])
                    src_doc.append(example['old_comment'])
                    dst_doc.append(example['new_comment'])
            sample = json.loads(line)
            if dataset=="Hebcup":
                cur_src_method = sample['src_method']
                cur_dst_method = sample['dst_method']
                cur_src_doc = sample['src_desc']
            else:
                cur_src_method = sample['old_code']
                cur_dst_method = sample['new_code']
                cur_src_doc = sample['old_comment']
            messages = [{"role": "system", "content": "You are a programmer who makes the code changes below:"}]
            head = "//write a docstring for after-change code based on the given before-change code and before-change docstring\n"
            prompt = ""
            if shots > 0:
                for srcm, dstm, srcd, dstd in zip(src_method, dst_method, src_doc, dst_doc):
                    prompt += head
                    prompt += f"//before-change code:\n{srcm}\n//before-change doc:\n{srcd}\n//after-change code:\n{dstm}\n//after-change docstring:\n{dstd}\nEND_OF_DEMO\n\n"
            prompt += head
            prompt += f"//before-change code:\n{cur_src_method}\n//before-change docstring:\n{cur_src_doc}\n//after-change code:\n{cur_dst_method}\n//after-change docstring:"
            # prompt = f"//write a docstring for after-change code based on the given before-change code and before-change docstring\n//before-change code:\n{src_method}\n//before-change docstring:\n{src_doc}\n//after-change code:\n{dst_method}\n//after-change docstring:"
            messages += [{"role": "user", "content": prompt}]
            if model=="gpt3.5":
                ret = collect_one_gpt(messages, api_key,temperature)
            else :
                ret = collect_one(messages, api_key,temperature,model)
            # post-process
            ret = process_line(ret)
            with jsonlines.open(out_path, "a") as fw:
                fw.write(ret)

                
def collect_all_retrieval_hybrid(dataset,model,train_path, test_path, out_path, shots, api_key, start,temperature):
    train = []
    with open(train_path, encoding="utf-8") as fr:
        for line in fr.readlines():
            train.append(json.loads(line))

    with open(f"ComSync/retrieval/{dataset}/dense_id_sampled.pkl", "rb") as fr:
        dense_rtr_ids = pickle.load(fr)
    with open(f"ComSync/retrieval/{dataset}/expert_id_sampled.pkl", "rb") as fr:
        expert_rtr_ids = pickle.load(fr)

    with open(test_path, encoding="utf-8") as fr:
        count = 0
        for _, (line, example1_ids,example2_ids) in tqdm(enumerate(zip(fr.readlines(), dense_rtr_ids,expert_rtr_ids)), total=len(dense_rtr_ids)):
            
            count += 1
            if count <= start:
                continue
            src_method = []
            dst_method = []
            src_doc = []
            dst_doc = []
            
            for id in example1_ids[:shots]:
                example = train[id]
                if dataset=="Hebcup":
                    src_method.append(example['src_method'])
                    dst_method.append(example['dst_method'])
                    src_doc.append(example['src_desc'])
                    dst_doc.append(example['dst_desc'])
                else:
                    src_method.append(example['old_code'])
                    dst_method.append(example['new_code'])
                    src_doc.append(example['old_comment'])
                    dst_doc.append(example['new_comment'])
            for id in example2_ids[:shots]:
                example = train[id]
                if dataset=="Hebcup":
                    src_method.append(example['src_method'])
                    dst_method.append(example['dst_method'])
                    src_doc.append(example['src_desc'])
                    dst_doc.append(example['dst_desc'])
                else:
                    src_method.append(example['old_code'])
                    dst_method.append(example['new_code'])
                    src_doc.append(example['old_comment'])
                    dst_doc.append(example['new_comment'])

            sample = json.loads(line)

            if dataset=="Hebcup":
                cur_src_method = sample['src_method']
                cur_dst_method = sample['dst_method']
                cur_src_doc = sample['src_desc']
            else:
                cur_src_method = sample['old_code']
                cur_dst_method = sample['new_code']
                cur_src_doc = sample['old_comment']
            messages = [{"role": "system", "content": "You are a programmer who makes the code changes below:"}]
            head = "//write a docstring for after-change code based on the given before-change code and before-change docstring\n"
            prompt = ""
            if shots > 0:
                for srcm, dstm, srcd, dstd in zip(src_method, dst_method, src_doc, dst_doc):
                    prompt += head
                    prompt += f"//before-change code:\n{srcm}\n//before-change doc:\n{srcd}\n//after-change code:\n{dstm}\n//after-change docstring:\n{dstd}\nEND_OF_DEMO\n\n"
            prompt += head
            prompt += f"//before-change code:\n{cur_src_method}\n//before-change docstring:\n{cur_src_doc}\n//after-change code:\n{cur_dst_method}\n//after-change docstring:"
            # prompt = f"//write a docstring for after-change code based on the given before-change code and before-change docstring\n//before-change code:\n{src_method}\n//before-change docstring:\n{src_doc}\n//after-change code:\n{dst_method}\n//after-change docstring:"
            messages += [{"role": "user", "content": prompt}]
            if model=="gpt3.5":
                ret = collect_one_gpt(messages, api_key,temperature)
            else :
                ret = collect_one(messages, api_key,temperature)
            # post-process
            ret = process_line(ret)
            with jsonlines.open(out_path, "a") as fw:
                fw.write(ret)

import argparse

if __name__ == "__main__":
    # TODO: USE WHICH IDS
    # todo start from 0
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--model", default="", type=str, required=True,
                        help="")
    parser.add_argument("--shot", default="", type=int, required=True,
                        help="")
    parser.add_argument("--api_key", default="", type=str, required=True,
                        help="")
    parser.add_argument("--dataset", default="", type=str, required=True,
                        help="")
    parser.add_argument("--retrieval", default="", type=str, required=True,
                        help="")
    parser.add_argument("--num", default="", type=int, required=False,
                        help="")
    args = parser.parse_args()
    #----------------------------------------------------------------#
    shot=args.shot
    model=args.model
    API_KEY = args.api_key
    dataset=args.dataset 
    retrieval=args.retrieval
    num=args.num
    #----------------------------------------------------------------#
    print(shot,model,API_KEY,dataset,retrieval,num)
    # rtr_path = f"./retrieval/{dataset}/{retrieval}_id_sampled.pkl"
    # train_path=f"./dataset/{dataset}/train.jsonl"
    # test_path =f"./dataset/{dataset}/test_sampled.jsonl"
    # if model!="gpt3.5":
    #     out_path = f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}_{num}.jsonl"
    # else:
    #     out_path = f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}.jsonl"
    # if retrieval!="hybrid":
    #     collect_all_retrieval(dataset,model,train_path,rtr_path, test_path, out_path, shots=shot, api_key=API_KEY, start=0,temperature=0.80)
    # else:
    #     collect_all_retrieval_hybrid(dataset,model,train_path,test_path, out_path, shots=shot, api_key=API_KEY, start=0,temperature=0.80)