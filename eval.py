# encoding=utf-8

"""
Evaluate results and calculate metrics

Usage:
    eval.py [options] TEST_SET RESULT_FILE DATASET

Options:
    -h --help                   show this screen.
    --metrics=<arg...>          metrics to calculate [default: accuracy,recall,distance]
    --eval-class=<str>          the class used to evaluate [default: Evaluator]
    --out-file=<str>            specify the output file path of eval. [default:]
    --dataset=<str>       
"""
import json
import logging
from utils.common import *
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple
from docopt import docopt
from utils.common import word_level_edit_distance
from utils.tokenizer import Tokenizer
from collections import Counter
from tokenizer import Tokenizer
# from nltk.translate.gleu_score import sentence_gleu
import jsonlines
stripAll = re.compile('[\s]+')
logging.basicConfig(level=logging.INFO)
EMPTY_TOKEN = '<empty>'


class BaseMetric(ABC):
    @staticmethod
    def isEqual(pred, oracle):
        predStr = stripAll.sub(' ', " ".join(pred).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        oracleStr = stripAll.sub(' ', " ".join(oracle).replace("<con>", '')).strip(' !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_')
        
        predStr = "".join([x for x in predStr if x.isalnum()])
        oracleStr = "".join([x for x in oracleStr if x.isalnum()])
       
        
        if predStr.lower() == oracleStr.lower():
            #print(predStr,"+",oracleStr)
            return True
        else:
            return False

class Accuracy(BaseMetric):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.correct_count = 0

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], src_references: Iterable[List[str]],
             out_file=None) -> dict:
        total = 0
        correct = 0
        item_with_labels = []
        for hypo_list, ref in zip(hypos, references):
            hypo = hypo_list[0]
            if not hypo:
                hypo = [EMPTY_TOKEN]
            assert (type(hypo[0]) == str)
            assert (type(ref[0]) == str)
            total += 1
            if self.isEqual(hypo, ref):
                item_with_labels.append(1)
                correct += 1
            else:
                item_with_labels.append(0)
        #print(total)
        if out_file is not None:
            with jsonlines.open(f"{out_file}/acc.jsonl", "w") as writer:
                writer.write_all(item_with_labels)
        # else:
        #     for hypo_list, ref in zip(hypos, references):
        #         hypo = hypo_list[0]
        #         if not hypo:
        #             hypo = [EMPTY_TOKEN]
        #         assert (type(hypo[0]) == str)
        #         assert (type(ref[0]) == str)
        #         total += 1
        #         if self.isEqual(hypo, ref):
        #             correct += 1
        return {'accuracy': correct / total, 'correct_count': correct}


class Recall(BaseMetric):
    def __init__(self, k: int = 5):
        super(Recall, self).__init__()
        self.k = k

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]], src_references: Iterable[List[str]],
             out_file=None) -> float:
        total = 0
        correct = 0
        item_with_labels = []
        for hypo_list, ref in zip(hypos, references):
            total += 1
            equal = False
            for hypo in hypo_list[:self.k]:
                if self.isEqual(hypo, ref):
                    item_with_labels.append(1)
                    equal = True
                    correct += 1
                    break
            if equal is False:
                item_with_labels.append(0)
        if out_file is not None:
            with jsonlines.open(f"{out_file}/recall.jsonl", "w") as writer:
                writer.write_all(item_with_labels)
        # else:
        #     for hypo_list, ref in zip(hypos, references):
        #         total += 1
        #         for hypo in hypo_list[: self.k]:
        #             if self.isEqual(hypo, ref):
        #                 correct += 1
        #                 break
        return correct / total


class EditDistance(BaseMetric):
    def __init__(self):
        super(EditDistance, self).__init__()

    @staticmethod
    def edit_distance(sent1: List[str], sent2: List[str]) -> int:
        return word_level_edit_distance(sent1, sent2)

    def relative_distance(self, src_ref_dis, hypo_ref_dis):
        if src_ref_dis == 0:
            logging.error("src_ref is the same as ref.")
            src_ref_dis = 1
       
        return hypo_ref_dis / src_ref_dis

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], out_file=None) -> dict:
        src_distances = []
        hypo_distances = []
        edit_diffs = []
        rel_distances = []
        ng_hypo_src_dis_lst = []
        ng_ref_src_dis_lst = []
        for idx, (hypo_list, ref, src_ref) in enumerate(zip(hypos, references, src_references)):
            hypo = " ".join(hypo_list[0]).replace("<con> ", "").lower().strip(".").split()
            src_ref = " ".join(src_ref).replace("<con> ", "").lower().strip(".").split()
            ref = " ".join(ref).replace("<con> ", "").lower().strip(".").split()
            hypo_ref_dis = self.edit_distance(hypo, ref)
            src_ref_dis = self.edit_distance(src_ref, ref)
            rel_distances.append(self.relative_distance(src_ref_dis, hypo_ref_dis))
            src_distances.append(src_ref_dis)
            hypo_distances.append(hypo_ref_dis)
            sign = int(np.sign(hypo_ref_dis - src_ref_dis))
            edit_diffs.append(sign)
            if sign > 0:
                hypo_src_dis = self.edit_distance(hypo, src_ref)
                
                ng_hypo_src_dis_lst.append(hypo_src_dis)
                ng_ref_src_dis_lst.append(src_ref_dis)
        if out_file is not None:
            with jsonlines.open(f"{out_file}/edit_dist.jsonl", "w") as fw:
                fw.write_all(edit_diffs)
        src_dis = float(np.mean(src_distances))
        hypo_dis = float(np.mean(hypo_distances))
        rel_dis = float(np.mean(rel_distances))
        
        ng_hypo_src_dis = float(np.mean(ng_hypo_src_dis_lst))
        ng_ref_src_dis = float(np.mean(ng_ref_src_dis_lst))
        print("ng_hypo_src_dis: ", ng_hypo_src_dis)
        print("ng_ref_src_dis: ", ng_ref_src_dis)
        print(Counter(edit_diffs))
        dist_reduced_rate = Counter(edit_diffs)[-1] / len(edit_diffs)
        dist_increased_rate = Counter(edit_diffs)[1] / len(edit_diffs)
        return {"rel_distance": rel_dis, "hypo_distance": hypo_dis, "src_distance": src_dis,
                "dist_reduced_rate": dist_reduced_rate, "dist_increased_rate": dist_increased_rate}




class BaseEvaluator(ABC):
    @abstractmethod
    def load_hypos_and_refs(self) -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        pass


class Evaluator(BaseEvaluator):
    METRIC_MAP = {
        "accuracy": Accuracy(),
        "recall": Recall(k=5),
        "distance": EditDistance(),
    }

    def __init__(self, args: dict, metric_map: dict = None):
        self.args = args
        self.metric_map = metric_map if metric_map else self.METRIC_MAP

    def load_hypos(self) -> List[List[List[str]]]:
        hypos = []
        with open(self.args['RESULT_FILE'], 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = [Tokenizer.tokenize_desc_with_con(item) for item in json.loads(line)]
                hypos.append(line)
        return hypos

    def load_refs(self):
        src_desc_lst = []
        dst_desc_lst = []
        with open(self.args['TEST_SET'], "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = json.loads(line)
                if self.args['DATASET']=="Hebcup":
                    src_desc_lst.append(line['src_desc_tokens'])
                    dst_desc_lst.append(line['dst_desc_tokens'])
                else:
                    src_desc_lst.append(line['old_comment_tokens'])
                    dst_desc_lst.append(line['new_comment_tokens'])
        return src_desc_lst, dst_desc_lst

    @staticmethod
    def normalize_hypos(hypos, src_references):
        new_hypos = []
        for hypo_list, src_sent in zip(hypos, src_references):
            if not hypo_list:
                print("find empty hypo list")
                hypo_list = [src_sent]
            new_hypos.append(hypo_list)
        return new_hypos

    def load_hypos_and_refs(self):
        src_references, references = self.load_refs()
        hypos = self.load_hypos()
        hypos = self.normalize_hypos(hypos, src_references)
        return hypos, references, src_references

    def cal_metrics(self, metrics: Iterable[str], hypos: List[List[List[str]]], references: List[List[str]],
                    src_references: List[List[str]], out_file):
        results = {}
        for metric in metrics:
            instance = self.metric_map[metric.lower()]
            
            results[metric] = instance.eval(hypos, references, src_references, out_file)
           
        return results

    def evaluate(self):
        metrics = self.args['--metrics'].split(',')
        out_file = self.args['--out-file']
        hypos, references, src_references = self.load_hypos_and_refs()
        assert type(hypos[0][0]) == type(references[0])

        results = self.cal_metrics(metrics, hypos, references, src_references, out_file)
        logging.info(results)
        lemma_results = {}
        return results, lemma_results

def evaluate(args, no_lemma=True):
    EvalClass = globals()[args['--eval-class']]
    evaluator = EvalClass(args)
    return evaluator.evaluate()


def main():
    args = docopt(__doc__)
    print(args)
    evaluate(args, True)


if __name__ == '__main__':
    main()
