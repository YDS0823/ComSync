DATASET="Hebcup"
TEST_SET="./dataset/"${DATASET}"/test_sampled.jsonl"
OUTPUT_FILE="./result/Hebcup/llama3_70b/shot8/llama3_70b_shot8_hybrid.jsonl" #your result

python eval.py ${TEST_SET}  ${OUTPUT_FILE}  ${DATASET}