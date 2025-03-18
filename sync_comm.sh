SHOT=2  # Number of shots seen by the model
MODEL="llama-70b"  # Select model from ["llama3-70b", "llama3-8b", "gpt3.5"]
API_KEY="your_api_key"  # Replace with actual API key
DATASET="Hebcup"  # Select dataset from ["Hebcup", "Panthap"]
RETRIEVAL="dense"  # Select retrieval way from ["dense", "expert", "random", "hybrid"]
NUM=1  # 1 to 10 for llama3 series, can be removed for gpt models

if [[ "$MODEL" == "gpt3.5" ]]; then
    python sync_comm.py  --model ${MODEL} --shot ${SHOT} --api_key ${API_KEY} --dataset ${DATASET}  --retrieval ${RETRIEVAL}
else
    python sync_comm.py  --model ${MODEL} --shot ${SHOT} --api_key ${API_KEY} --dataset ${DATASET}  --retrieval ${RETRIEVAL}  --num ${NUM}
fi
