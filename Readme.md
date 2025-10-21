## Readme

Download the R<sup>2</sup>ComSync dataset from the download [link](https://zenodo.org/records/17362180) and unzip it to the dataset folder

### Retrieval

The samples we retrieved are from the training sets of the respective datasets.
Retrieve 10 relevant training set samples for each test sample:

```shell
python retrieval.py  --dataset Hebcup
```

### Prompt

Here we present a prompt used for one sample from the **Liu’s dataset** test set. Under the **EHR** condition, the number of selected shots is **2**.

```shell
{"role": "system", "content": "You are a programmer who makes the code changes below:"}
{"role": "user", "content": "
//write a docstring for after-change code based on the given before-change code and before-change docstring.End your answer with 
END_OF_DEMO

//before-change code:
public void removeRequestFilter(Class<?> requestFilterClass) {\n     msf4JRequestFilterListMap.remove(requestFilterClass);\n     updateGlobalRequestFilterSet(); // Update global filter class set\n   }
//before-change docstring:
public void removeRequestInterceptor(boolean isGlobal, MSF4JRequestInterceptor requestInterceptor) {\n     if (isGlobal) {\n       globalRequestInterceptorList.remove(requestInterceptor);\n     }\n     requestInterceptorMap.remove(requestInterceptor.getClass());\n   }
//after-change code:
Remove msf4j request filter.
//after-change docstring:
Remove msf4j request interceptor.
END_OF_DEMO

//before-change code:
public static String toString(List<String> strings) {\n     return toString(strings, Const.EOL);\n   }
//before-change docstring:
public static <T> String toString(List<T> list) {\n     return toString(list, Const.EOL);\n   }
//after-change code:
Concatenates a list of strings to a single string, separated by line breaks.
//after-change docstring:
Converts and concatenates a list of objects to a single string, separated by line breaks.
END_OF_DEMO

//before-change code:
public List<WanPublisherConfig> getWanPublisherConfigs() {\n     return wanPublisherConfigs;\n   }
//before-change docstring:
public @Nonnull\n   List<CustomWanPublisherConfig> getCustomPublisherConfigs() {\n     return customPublisherConfigs;\n   }
//after-change code:
Returns the list of configured WAN publisher targets for this WAN\nreplication.
//after-change docstring:
"
}
```
The first sample was retrieved using **CodeBERT**, with the following basic information:
`{"sample_id": 3880860, "full_name": "wso2/msf4j", "commit_id": "8df74e2f798af426820f34511492b3c27b96984b"}`

The second sample was retrieved using the **Expert vector**, with the following basic information:
`{"sample_id": 3654886, "full_name": "TEAMMATES/teammates", "commit_id": "efa4df2141589cc35ac62fc086481201fa9b6413"}`

The code synchronization annotation task needs to be performed on the following problem sample:
`{"sample_id": 2231080, "full_name": "hazelcast/hazelcast", "commit_id": "b4af08812320bcadbbd5cd8e496aa8e96993aaa3"}`

The answers provided by **Llama3-8b-instruct**, **GPT-3.5-turbo**, and **Llama3-70b-instruct** are as follows:

```shell
Llama3-8b-instruct：
["Returns the list of configured custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher configurations for this WAN replication.", "Returns the list of custom WAN publisher configurations for this WAN replication.", "Returns the list of custom configured WAN publisher targets for this WAN replication.", "Gets the list of custom WAN publisher configurations.", "Returns the list of custom WAN publisher configurations for this WAN replication.", "Returns the list of configured custom WAN publisher targets for this WAN replication.", "Get the list of configured custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher configurations for this WAN replication.", "Returns the list of custom WAN publisher configurations."]
Llama3-70b-instruct：
["Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication.", "Returns the list of custom WAN publisher targets for this WAN replication."]
GPT-3.5-turbo：
["Get the list of custom WAN publisher configurations.", "Get the list of custom publisher configurations for this WAN replication.", "Get the list of custom WAN publisher configurations for this WAN replication.", "Returns the list of custom WAN publisher configurations for this WAN replication.", "Returns the list of custom configured WAN publisher targets for this WAN replication.", "Get the list of custom configured WAN publisher targets for this WAN replication.", "Get the list of custom WAN publisher configurations configured for this WAN replication."]
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

