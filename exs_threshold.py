import json
from tokenizer import Tokenizer
from utils.common import word_level_edit_distance
from tqdm import tqdm
import re
import numpy as np
from collections import defaultdict

def exsessive_edit(src_desc_tokens, item_i):
    dist = word_level_edit_distance(src_desc_tokens, item_i)
    ratio = dist / len(src_desc_tokens)
    return ratio


bins = np.linspace(0, 1, num=21)  
bin_labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]

exs_ratio_counts = defaultdict(int)

failure=0
count=0
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
                exs_ratio = exsessive_edit(src_desc_tokens, i)
                if exs_ratio>=1:
                     cnt+=1
                bin_index = np.digitize(exs_ratio, bins) - 1  
                if 0 <= bin_index < len(bin_labels):
                    exs_ratio_counts[bin_labels[bin_index]] += 1
            else:
                failure+=1

sum=0
for range_key in bin_labels:
    sum+=exs_ratio_counts[range_key]
    print(f"{range_key}: {exs_ratio_counts[range_key]},{sum/count}")
print(failure,cnt)