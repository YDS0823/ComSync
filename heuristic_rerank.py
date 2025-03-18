import json
import jsonlines
from tokenizer import Tokenizer
from utils.common import word_level_edit_distance
from tqdm import tqdm
import re
from utils.edit import DiffTokenizer, empty_token_filter, construct_diff_sequence_with_con

def get_func_name(text):
    pattern = re.compile(
        r"(public|private|static|protected|abstract|native|synchronized)*\s.+\(.*\)\s*(throws .+)?\s*\{",
        flags=re.DOTALL)
    text_lst = text.strip().split("\n")
    id = 0
    for i in text_lst:
        if i.strip().startswith('@') and not pattern.search(i):
            id += 1
        else:
            break
    text = "\n".join(text_lst[id:])
    s, e = pattern.search(text).span()
    head = text[s:e]
    pattern1 = re.compile("([a-zA-Z0-9_]+)\s*\(")
    s1, e1 = pattern1.search(head).span()
    return head[s1:e1-1].strip()


def is_update_func_name(old_code, new_code, old_desc, new_desc):
    old_name = get_func_name(old_code)
    new_name = get_func_name(new_code)
    if old_name != new_name:
        diff_tokenizer = DiffTokenizer(token_filter=empty_token_filter)
        old_name_tokens, new_name_tokens = diff_tokenizer(old_name, new_name)
        name_change_seqs = construct_diff_sequence_with_con(old_name_tokens,
                                                       new_name_tokens)
        old_desc_tokens, new_desc_tokens = diff_tokenizer(old_desc, new_desc)
        desc_change_seqs = construct_diff_sequence_with_con(old_desc_tokens,
                                                       new_desc_tokens)
        new_desc_change_seqs = []

        for k in desc_change_seqs:
            if k[0].lower() == k[1].lower():
                k[2] = 'equal'
            new_desc_change_seqs.append([m.lower() for m in k])
        new_name_change_seqs = []
        for k in name_change_seqs:
            if k[0].lower() == k[1].lower():
                k[2] = 'equal'
            new_name_change_seqs.append([m.lower() for m in k])
        replace_lst = []
        for i in new_name_change_seqs:
            if i[2] == 'replace':
                replace_lst.append(i)
        for j in replace_lst:
            if j not in new_desc_change_seqs:
                return False
        return True
    else:
        return True

def unknow_tokens(src_desc_tokens, item_i):
    unknown_token_ct = 0
    for i in item_i:
        if i not in src_desc_tokens:
            unknown_token_ct += 1
    return unknown_token_ct / len(item_i)

def exsessive_edit(src_desc_tokens, item_i):
    dist = word_level_edit_distance(src_desc_tokens, item_i)
    ratio = dist / len(src_desc_tokens)
    return ratio

def rerank(path, unk_threshold, exs_threshold,dataset):
    def rule_one(ret, line):
        new_ret = []
        bad_ret = []
        for item in ret:
            if dataset=="Hebcup":
                is_update = is_update_func_name(line['src_method'], line['dst_method'], line['src_desc'], item) 
            else:
                is_update = is_update_func_name(line['old_code'], line['new_code'], line['old_comment'], item)
            if not is_update:
                bad_ret.append(item)
            else:
                new_ret.append(item)
        ret = new_ret + bad_ret
        return ret

    def rule_two(ret, src_desc_tokens, unk_threshold):
        if unk_threshold is not None:
            new_ret = []
            bad_ret = []
            for item in ret:
                stripAll = re.compile('[\s]+')
                tmp = stripAll.sub(' ', item).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
                tmp = "".join([x for x in tmp if x.isalnum()])
                if tmp != "":
                    i = Tokenizer.tokenize_desc_with_con(item.lower())
                    while "<con>" in i:
                        i.remove("<con>")
                    unk_ratio = unknow_tokens(src_desc_tokens, i)
                    if unk_ratio < unk_threshold:
                        new_ret.append(item)
                    else:
                        bad_ret.append(item)
                else:
                    bad_ret.append(item)
            ret = new_ret + bad_ret
        return ret

    def rule_three(ret, src_desc_tokens, exs_threshold):
        if exs_threshold is not None:
            new_ret = []
            bad_ret = []
            for item in ret:
                stripAll = re.compile('[\s]+')
                tmp = stripAll.sub(' ', item).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
                tmp = "".join([x for x in tmp if x.isalnum()])
                if tmp != "":
                    i = Tokenizer.tokenize_desc_with_con(item.lower())
                    while "<con>" in i:
                        i.remove("<con>")
                    exs_ratio = exsessive_edit(src_desc_tokens, i)
                    if exs_ratio < exs_threshold:
                        new_ret.append(item)
                    else:
                        bad_ret.append(item)
                else:
                    bad_ret.append(item)
            ret = new_ret + bad_ret
        return ret

    results = []
    with open(path, encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            results.append(line)
    new_results = []
    count = 0
    # with open("./dataset/test_clean.jsonl", encoding="utf-8") as fr:
    with open(f"./dataset/{dataset}/test_sampled.jsonl", encoding="utf-8") as fr:
        for ret, line in tqdm(zip(results, fr.readlines()), total=len(results)):
            count += 1
            line = json.loads(line)
            if dataset=="Hebcup":
                src_desc_tokens = Tokenizer.tokenize_desc_with_con(line['dst_desc'].lower())
            else:
                src_desc_tokens = Tokenizer.tokenize_desc_with_con(line['old_comment'].lower())
            while "<con>" in src_desc_tokens:
                src_desc_tokens.remove("<con>")
            ret = rule_one(ret, line)
            ret = rule_two(ret, src_desc_tokens, unk_threshold)
            ret = rule_three(ret, src_desc_tokens, exs_threshold)

            new_results.append(ret)
    with jsonlines.open(path+f"_unk{unk_threshold}_exs{exs_threshold}.jsonl", "w") as fw:
        fw.write_all(new_results)

import argparse

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--dataset", default="", type=str, required=True,
                        help="choose a dataset")
    parser.add_argument("--unk_threshold", default="", type=float, required=True,
                        help="")
    parser.add_argument("--exs_threshold", default="", type=float, required=True,
                        help="")
    parser.add_argument("--result_path", default="", type=str, required=True,
                        help="")
    args = parser.parse_args()
    dataset=args.dataset
    unk_threshold=args.unk_threshold
    exs_threshold=args.exs_threshold
    path=args.result_path
    print(dataset,unk_threshold,exs_threshold,path)
    rerank(path,unk_threshold, exs_threshold,dataset)

