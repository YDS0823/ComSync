from utils.common import remove_comm
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import json
import pickle

def construct_dense_dt(dataset_path, cuda,dataset,output_path):
    embeds = []
    device = torch.device("cuda", cuda)

    with open(dataset_path, encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()):
            line = json.loads(line)
            if dataset=="hebcup":
                src_method = remove_comm(line['src_method'])
                dst_method = remove_comm(line['dst_method'])
                src_desc = line['src_desc']
            else:
                src_method = remove_comm(line['old_code'])
                dst_method = remove_comm(line['new_code'])
                src_desc = line['old_comment']

            tok = AutoTokenizer.from_pretrained("codebert-base")
            encoder = AutoModel.from_pretrained("codebert-base").to(device)
            src_method_toks = tok.tokenize(src_method)
            dst_method_toks = tok.tokenize(dst_method)
            src_desc_toks = tok.tokenize(src_desc)
            src_method_ids = tok.convert_tokens_to_ids(src_method_toks)
            dst_method_ids = tok.convert_tokens_to_ids(dst_method_toks)
            src_desc_ids = tok.convert_tokens_to_ids(src_desc_toks)
            cls_id = tok.convert_tokens_to_ids(tok.cls_token)
            eos_id = tok.convert_tokens_to_ids(tok.eos_token)

            if len(src_method_ids) > 510:
                src_method_ids = src_method_ids[:510]
            if len(dst_method_ids) > 510:
                dst_method_ids = dst_method_ids[:510]
            if len(src_desc_ids) > 510:
                src_desc_ids = src_desc_ids[:510]
            src_method_embed = encoder(torch.tensor([cls_id] + src_method_ids + [eos_id], device=device)[None, :])[0][:, 0, :]
            dst_method_embed = encoder(torch.tensor([cls_id] + dst_method_ids + [eos_id], device=device)[None, :])[0][:, 0, :]
            src_desc_embed = encoder(torch.tensor([cls_id] + src_desc_ids + [eos_id], device=device)[None, :])[0][:, 0, :]
            embedding = src_method_embed + dst_method_embed + src_desc_embed
            embedding_cpu = embedding[0].detach().cpu().numpy().tolist()
            embeds.append(embedding_cpu)

    with open(output_path, "wb") as fw:
        pickle.dump(embeds, fw)
if __name__=="__main__":
    dataset="Hebcup"
    mode="test"
    input_path="./dataset/"+dataset+f"/{mode}.jsonl"
    output_path="./dataset/"+dataset+f"Codebert/{mode}_dense_DB.pkl"
    construct_dense_dt(input_path,1,dataset,output_path)
