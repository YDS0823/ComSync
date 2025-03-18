#!/usr/bin/env python
# encoding: utf-8
'''
@author: YANG Zhen
@contact: zhyang8-c@my.cityu.edu.hk
@file: concat_rets.py
@time: 3/14/2023 10:26 AM
@desc:
'''

import os
import json
import jsonlines
import re


def process_line(line):
    new_line = []
    for sample in line:
        #替换所有连续的空白字符为单个空格
        sample = re.sub(r"\s+", " ", sample)
        #将电子邮件地址替换为 "EMAIL"。
        sample = re.sub(r"^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$", "EMAIL", sample)
        #将 URL 替换为 "URL"
        sample = re.sub(r"[a-zA-z]+://[^\s]*", "URL", sample)
        #将版本号替换为 "version VERSION"
        sample = re.sub(r"version [0-9]+.[0-9]+.[0-9]+", "version VERSION", sample)
        found = False
        for sw in ["END_OF_DEMO", "END_OF_DOC", "//before-change code:",
                   "//after-change code:", "//before-change docstring:", "after-change docstring"]:
            if sw in sample:
                sample = sample.split(sw)[0].strip()
                if sample != "":
                    new_line.append(sample)
                found = True
                break
            else:
                sample_lst = sample.split()
                last_token = sample_lst[-1]
                if last_token in sw:
                    sample = " ".join(sample_lst[:-1]).strip()
                    if sample != "":
                        new_line.append(sample)
                    found = True
                    break
        if not found:
            sample = re.split(r"```|//|>>>|/\*", sample)[0].strip()
            if sample != "":
                new_line.append(sample)
    return new_line

def post_process(file):
    new_results = []
    with open(f"../results/{file}", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            new_line = process_line(line)
            new_results.append(new_line)
    with jsonlines.open(f"../{file}", "w") as fw:
        fw.write_all(new_results)


def concat_hybrid():
    # TODO: root directory
    root = "../result_hybrid_ids_shot=8"
    ids = []
    for i in range(9204):
        if i % 499 == 0:
           ids.append(i)

    ids.append(9203)
    hybrid_rets = []
    print("-------")
    for j in ids:
        with open(f"{root}/results_rtr_hybrid_ids{j}.jsonl", "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = json.loads(line)
                line = process_line(line)
                hybrid_rets.append(line)

    # TODO: output file
    with jsonlines.open("../results/results_rtr_hybrid_ids_shot8.jsonl", "w") as fw:
        fw.write_all(hybrid_rets)

def concat_dense():
    root = "../result_dense_ids(4491-9203)"
    dense_rets = []
    with open(f"{root}/results_rtr_dense_ids.jsonl", encoding="utf-8") as fr:
        for i, line in enumerate(fr.readlines()):
            if i == 4492:
                break
            line = json.loads(line)
            line = process_line(line)
            dense_rets.append(line)
    ids = []
    for i in range(4990, 9204):
        if i % 499 == 0:
            ids.append(i)
    ids.append(9203)

    for j in ids:
        with open(f"{root}/results_rtr_dense_ids{j}.jsonl", "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = json.loads(line)
                line = process_line(line)
                dense_rets.append(line)

    with jsonlines.open("../results/results_rtr_dense_ids_shot3.jsonl", "w") as fw:
        fw.write_all(dense_rets)

if __name__ == "__main__":
    # concat_hybrid()
    post_process("results_rtr_hybrid_ids_shot8.jsonl")

