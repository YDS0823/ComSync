import os
import time
import json
import pickle
import jsonlines
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils.concat_rets import process_line


# ====================================================
# =============== 核心函数定义 =========================
# ====================================================

def build_prompt(messages, tokenizer):
    """构造完整 prompt"""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def safe_prompt(prompt_text, tokenizer, max_len=32000):
    """安全截断 prompt"""
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(ids) > max_len:
        print("⚠️ Prompt 超过最大长度，自动截断")
        ids = ids[-max_len:]
    return tokenizer.decode(ids)


def load_train(train_path):
    with open(train_path, "r", encoding="utf-8") as fr:
        return [json.loads(line) for line in fr]


def generate_with_vllm(llm, tokenizer, prompts, sampling_params, out_path, process_line_fn):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = jsonlines.open(out_path, mode="a")

    t_start = time.time()
    for i in tqdm(range(0, len(prompts)), desc="Batches"):
        batch_prompts = [prompts[i]]
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)

        for final_output, prompt_j in zip(outputs, batch_prompts):
            try:
                cand_texts = [o.text for o in final_output.outputs]
            except Exception:
                cand_texts = [str(final_output)]

            candidates = []
            for text in cand_texts:
                if text.startswith(prompt_j):
                    gen_text = text[len(prompt_j):].strip()
                else:
                    gen_text = text.strip()
                if "//after-change docstring:" in gen_text:
                    candi = gen_text.split("//after-change docstring:")[-1].strip()
                else:
                    candi = gen_text.strip()
                candi = candi.lstrip()
                if candi.lower().startswith("assistant"):
                    candi = candi[len("assistant"):].lstrip(":： \n")
                if "END_OF_DEMO" in candi:
                    candi = candi.split("END_OF_DEMO")[0].strip()
                candidates.append(candi)

            ret = process_line_fn(candidates if candidates else ["//None"])
            writer.write(ret)
            writer._fp.flush()
            try:
                os.fsync(writer._fp.fileno())
            except Exception:
                pass

    writer.close()
    total_time = time.time() - t_start
    print(f"✅ All done. Total time: {total_time:.2f}s, avg per item: {total_time/len(prompts):.2f}s")


def build_prompts(test_path, rtr_paths, train, tokenizer, shots, hybrid=False):
    """构造 prompts（支持 hybrid 与单一检索）"""
    prompts = []
    if hybrid:
        with open(rtr_paths[0], "rb") as fr1, open(rtr_paths[1], "rb") as fr2:
            rtr1_ids, rtr2_ids = pickle.load(fr1), pickle.load(fr2)
        iterator = zip(open(test_path, encoding="utf-8"), rtr1_ids, rtr2_ids)
    else:
        with open(rtr_paths[0], "rb") as fr:
            rtr_ids = pickle.load(fr)
        iterator = zip(open(test_path, encoding="utf-8"), rtr_ids)

    for line_pack in iterator:
        sample = json.loads(line_pack[0])
        src_method, dst_method, src_doc, dst_doc = [], [], [], []

        if hybrid:
            example1_ids, example2_ids = line_pack[1], line_pack[2]
            for idx1, idx2 in zip(example1_ids[:shots], example2_ids[:shots]):
                for idx in [idx1, idx2]:
                    ex = train[idx]
                    src_method.append(ex['src_method'])
                    dst_method.append(ex['dst_method'])
                    src_doc.append(ex['src_desc'])
                    dst_doc.append(ex['dst_desc'])
        else:
            example_ids = line_pack[1]
            for idx in example_ids[:shots]:
                ex = train[idx]
                src_method.append(ex['src_method'])
                dst_method.append(ex['dst_method'])
                src_doc.append(ex['src_desc'])
                dst_doc.append(ex['dst_desc'])

        cur_src_method = sample['src_method']
        cur_dst_method = sample['dst_method']
        cur_src_doc = sample['src_desc']

        messages = [{"role": "system", "content": "You are a programmer who makes the code changes below:"}]
        head = "//write a docstring for after-change code based on the given before-change code and before-change docstring.End your answer with \nEND_OF_DEMO\n\n"

        prompt_text = ""
        for sm, dm, sd, dd in zip(src_method, dst_method, src_doc, dst_doc):
            prompt_text += f"{head}//before-change code:\n{sm}\n//before-change docstring:\n{sd}\n//after-change code:\n{dm}\n//after-change docstring:\n{dd}\nEND_OF_DEMO\n\n"

        prompt_text += f"{head}//before-change code:\n{cur_src_method}\n//before-change docstring:\n{cur_src_doc}\n//after-change code:\n{cur_dst_method}\n//after-change docstring:"

        messages.append({"role": "user", "content": prompt_text})
        full_prompt = build_prompt(messages, tokenizer)
        full_prompt = safe_prompt(full_prompt, tokenizer)
        prompts.append(full_prompt)
    return prompts


# ====================================================
# =============== 主入口：parse + 主逻辑 ===============
# ====================================================

def main():
    parser = argparse.ArgumentParser(description="Unified vLLM Retrieval Generation Script")
    parser.add_argument("--model_path", required=True, help="本地模型路径")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--retrieval_paths", nargs="+", required=True, help="一个或两个检索路径（dense, expert）")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--hybrid", action="store_true", help="是否使用 dense+expert 混合检索")
    parser.add_argument("--max_tokens", type=int, default=50, help="生成长度（Python 可设 100）")

    args = parser.parse_args()

    print(f"🚀 Loading model from {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="float16",
        tensor_parallel_size=4,
        max_num_batched_tokens=16384,
        max_num_seqs=64,
        gpu_memory_utilization=0.95,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=10,
    )

    train = load_train(args.train_path)
    prompts = build_prompts(args.test_path, args.retrieval_paths, train, tokenizer, args.shots, hybrid=args.hybrid)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{'hybrid' if args.hybrid else 'single'}_shot{args.shots}.jsonl"
    )

    print(f"📝 Output will be saved to: {out_path}")
    generate_with_vllm(llm, tokenizer, prompts, sampling_params, out_path, process_line)


if __name__ == "__main__":
    main()
