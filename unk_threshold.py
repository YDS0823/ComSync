import json
from tokenizer import Tokenizer
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict

def unknow_tokens(src_desc_tokens, item_i):
    unknown_token_ct = 0
    for i in item_i:
        if i not in src_desc_tokens:
            unknown_token_ct += 1
    return unknown_token_ct / len(item_i)


bins = np.linspace(0, 1, num=21) 
bin_labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

unk_ratio_counts = defaultdict(int)
count=0
failure=0
cnt=0
#change to your dataset path
with open("./dataset/Hebcup/train.jsonl", encoding="utf-8") as fr:
    for line in tqdm(fr.readlines()):
        count+=1
        line = json.loads(line)
        src_desc_tokens = Tokenizer.tokenize_desc_with_con(line['src_desc'].lower())
        while "<con>" in src_desc_tokens:
            src_desc_tokens.remove("<con>")
        stripAll = re.compile('[\s]+')
        tmp = stripAll.sub(' ', line['dst_desc']).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        tmp = "".join([x for x in tmp if x.isalnum()])
        if tmp != "":
            i = Tokenizer.tokenize_desc_with_con(line['dst_desc'].lower())
            while "<con>" in i:
                i.remove("<con>")
            unk_ratio = unknow_tokens(src_desc_tokens, i)
            if unk_ratio==1:
                cnt+=1
            
            bin_index = np.digitize(unk_ratio, bins) - 1  
            if 0 <= bin_index < len(bin_labels):
                unk_ratio_counts[bin_labels[bin_index]] += 1
        else:
            failure+=1

sum=0
for range_key in bin_labels:
    sum+=unk_ratio_counts[range_key]
    print(f"{range_key}: {unk_ratio_counts[range_key]},{sum/count}")
print(failure,cnt)