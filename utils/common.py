# encoding=utf-8

import re
import random
import importlib
import numpy as np
from typing import List, Iterable
#动态规划，表示通过插入、删除和替换操作将一个字符串转换为另一个字符串所需的最少操作次数。
def word_level_edit_distance(a: List[str], b: List[str]) -> int:
    max_dis = max(len(a), len(b))
    distances = [[max_dis for j in range(len(b)+1)] for i in range(len(a)+1)]
    #创建了一个二维列表 distances，其中每个元素都被初始化为 max_dis。这个列表的行数是 a 列表的长度加1，列数是 b 列表的长度加1。
    #每行每列加上头部
    for i in range(len(a)+1):
        distances[i][0] = i
    for j in range(len(b)+1):
        distances[0][j] = j
    #遍历内容
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            distances[i][j] = min(distances[i-1][j] + 1,
                                  distances[i][j-1] + 1,
                                  distances[i-1][j-1] + cost)
    return distances[-1][-1]


def recover_desc(sent: Iterable[str]) -> str:
    return re.sub(r' <con> ', "", " ".join(sent).lower())

def remove_comm(program):
    pattern = re.compile(r"/\*.+?\*/", flags=re.DOTALL)
    program = re.sub(pattern, "", program)
    program = re.sub("// .+?\n", "", program)
    program = re.sub("\s+", " ", program)
    return program.strip()
