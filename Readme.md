## Readme

Download the dataset from the download [link](https://zenodo.org/records/17358768) and unzip it to the dataset folder

### Retrieval

Retrieve 100 relevant training set samples for each test sample:

```shell
python retrieval.py  --dataset Hebcup
```

### Generation

Enter sync_comm.sh to modify the required parameters:

```python
SHOT=2  # Number of shots seen by the model
MODEL="llama-70b"  # Select model from ["llama3-70b", "llama3-8b", "gpt3.5"]
API_KEY="your_api_key"  # Replace with actual API key
DATASET="Hebcup"  # Select dataset from ["Hebcup", "Panthap"]
RETRIEVAL="dense"  # Select retrieval way from ["dense", "expert", "random", "hybrid"]
NUM=1  # 1 to 10 for llama3 series, can be removed for gpt models
```

Then use sync_comm.py to call the large language model for generation:

```shell
bash sync_comm.sh
```

The results of llama3 need to be post processed and merged:

```shell
python post_process.py
```

### Rerank

Use heuristic_rerank.py to rerank and control the effect  changing `unk_threshold` and `exs_threshold`:

```shell
python heuristic_rerank.py --dataset Hebcup --unk_threshold 0.35 --exs_threshold 0.25 --result_path "./result/Hebcup/llama3_70b/shot8/llama3_70b_shot8_hybrid.jsonl"
```

You can use the following code to determine the `unk_threshold` and `exs_threshold` required for your dataset.

```shell
python unk_threshold.py
python exs_threshold.py
```

### Evaluation

Use eval.sh for evaluation:

```shell
DATASET="Hebcup" 
TEST_SET="./dataset/"${DATASET}"/test_sampled.jsonl"
OUTPUT_FILE="./result/Hebcup/llama3_70b/shot8/llama3_70b_shot8_hybrid.jsonl_unk0.35_exs0.25.jsonl" #your result

python eval.py ${TEST_SET}  ${OUTPUT_FILE}  ${DATASET}
```

