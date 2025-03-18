import re
import json

prefixes_to_remove = [
    r"Here is the rewritten docstring for the after-change code:",
    r"Here is the written docstring for the after-change code:",
    r"Here is the rewritten docstring:",
    r"Here is the written docstring for the after-change code based on the given before-change code and before-change docstring:",
    r"Here are the docstrings for the after-change code:",
    r"Based on the given before-change code and before-change docstring, I would write the following docstring for the after-change code:",
    r"Based on the given before-change code and before-change docstring, I will write the docstring for the after-change code. Here are the rewritten docstrings:",
    r"Here is the rewritten docstring based on the before-change code and before-change docstring:",
    r"Here is the docstring for the after-change code based on the given before-change code and before-change docstring:",
    r"Here is the docstring for the after-change code:",
    r"Based on the provided before-change code and docstrings, I will write a docstring for the after-change code.",
    r"Based on the before-change code and docstring, I will write a docstring for the after-change code.",
    r"Based on the given before-change code and before-change docstring, here is the written docstring for the after-change code:",
    r"Here is the rewritten docstring for the after-change code based on the given before-change code and before-change docstring:",
    r"Here is the rewritten docstring based on the before-change code and docstring:",
    r"Here are the rewritten docstrings for the after-change code:",
    r"Here are the rewritten docstrings for the after-change code based on the given before-change code and before-change docstring:",
    r"Here are the rewritten docstrings for the after-change code based on the given before-change code and before-change docstrings:",
    r"Here is the rewritten docstring based on the given before-change code and before-change docstring:",
    r"Here are the docstrings for the after-change code based on the given before-change code and before-change docstring:",
    r"Here are the docstrings for the after-change code based on the given before-change code and before-change docstrings:",
    r"Based on the before-change code and docstring, here is a rewritten docstring for the after-change code:",
    r"Based on the before-change code and before-change docstring,",
    r"Based on the provided before-change code and docstring, I will write a docstring for the after-change code.",
    r"Based on the provided before-change code and before-change docstrings, I will write a docstring for the after-change code.",
    r"Based on the given before-change code and before-change docstring,",
    r"Based on the before-change code and docstring, the after-change code is:",
    r"Based on the provided code snippets, I will write a docstring for each of the after-change code snippets based on the given before-change code and before-change docstrings.",
    r"Based on the provided before-change code and docstrings, I'll write a docstring for the after-change code. Here are the results:",
    r"Based on the before-change code and docstring, the after-change code and docstring are:",
    r"Based on the provided before-change code and before-change docstrings, I will write the docstrings for the after-change code.",
    r"Based on the before-change code and docstring, I will write a docstring for the after-change code:",
    r"Based on the provided before-change code and before-change docstring, I will write a docstring for the after-change code.",
    r"Based on the before-change code and docstring, here is a possible docstring for the after-change code:",
    r"Based on the provided before-change code and before-change docstring, here is the written docstring for the after-change code:",
    r"Based on the provided before-change code and before-change docstrings, I will write the after-change code and docstrings as follows:",
    r"I've written docstrings for the after-change code based on the given before-change code and before-change docstrings. Here are the results:",
    r"Based on the provided before-change code and docstring, I'll write a docstring for the after-change code.",
    r"Here is the after-change docstring:",
    r"Here is a possible docstring for the after-change code:",
    r"Here is the rewritten docstring based on the after-change code:",
    r"Here is the updated docstring for the after-change code:",
    r"The docstring for the after-change code would be:",
    r"Here is the rewritten docstring for the after-change code: ",
    r"Here is the written docstring for the after-change code: ",
    r"Here are the rewritten docstrings based on the given before-change code and before-change docstrings:",
    r"Here are the docstrings for each of the given examples:",
    r"Based on the given before-change code and before-change docstring, I will write a docstring for the after-change code.",
    r"Based on the given before-change code and before-change docstring, the after-change code is:",
    r"Based on the given before-change code and before-change docstring, I will write the docstring for the after-change code.",
    r"Here is the docstring:",
    r"Here is the after-change code:",
    r"The after-change code is:",
    r"The after-change code:",
    r"Here are the docstrings for each of the given code changes:",
    r"Based on the given before-change code and before-change docstring, here is the after-change docstring:",
    r"Based on the before-change code and docstring, here is the rewritten docstring for the after-change code:",
    r"The docstring for the after-change code is:",
    r"Based on the given before-change code and before-change docstring:",
    r"Based on the given before-change code and before-change docstring, the after-change docstring is:",
    r"Based on the given before-change code and before-change docstring, the after-change docstring for the after-change code is:",
]

pattern = re.compile('|'.join(prefixes_to_remove))

#post process
for dataset in ["Hebcup","Panthap"]:
    for shot in range(2,11,2):
        for model in ["llama3_8b","llama3_70b"]:
            for retrieval in ["dense","expert","hybrid"]:
                for num in range(1,11):
                    input_file = f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}_{num}.jsonl"
                    output_file= f"./result/{dataset}/{model}/shot{shot}/post_{model}_shot{shot}_{retrieval}_{num}.jsonl"
                    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                        for line in infile:
                            sentence = json.loads(line.strip())
                            if sentence:
                                processed_sentence = pattern.sub("", sentence[0]).strip()
                                output_sentence = [processed_sentence] if processed_sentence else [""] 
                            else:
                                output_sentence = [""]  
                            
                            json.dump(output_sentence, outfile)
                            outfile.write('\n')


#merge
for dataset in ["Hebcup","Panthap"]:
    for shot in range(2,11,2):
        for model in ["llama3_8b","llama3_70b"]:
            for retrieval in ["dense","expert","hybrid"]:
                input_file=f"./result/{dataset}/{model}/shot{shot}/post_{model}_shot{shot}_{retrieval}_1.jsonl"
                output_file=f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}.jsonl"
                with open(input_file,'r',encoding='utf-8') as infile, open(output_file,'w',encoding='utf-8') as outfile:
                    for line in infile:
                        outfile.write(line) 

for dataset in ["Hebcup","Panthap"]:
    for shot in range(2,11,2):
        for model in ["llama3_8b","llama3_70b"]:
            for retrieval in ["dense","expert","hybrid"]:
                for num in range(2,11):
                    with open(f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}.jsonl",'r') as file:
                        file1_lines=file.readlines()
                    with open(f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}_{num}.jsonl",'r')as file:
                        file2_lines=file.readlines()
                    if len(file1_lines) != len(file2_lines):
                        print(type,shot,num)
                        raise ValueError("number of line")
                    merged_lines=[]
                    for line1,line2 in zip(file1_lines,file2_lines):
                        data1=json.loads(line1)
                        data2=json.loads(line2)
                        merged_lines.append(json.dump(data1+data2))
                    with open(f"./result/{dataset}/{model}/shot{shot}/{model}_shot{shot}_{retrieval}.jsonl", 'w') as file:
                        for line in merged_lines:
                            file.write(line + '\n')

