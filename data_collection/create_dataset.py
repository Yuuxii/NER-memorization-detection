import pandas as pd
import json

with open("prompts/declarative_sentences.txt", encoding='utf-8') as f:
    prompts = f.read()
        # print(prompts)
    prompts = prompts.split("\n")

de_prompts = prompts[:100]
ex_prompts = prompts[100:200]
im_prompts = prompts[200:300]
in_prompts = prompts[200:400]
manual_prompts = prompts[400:]


dev_names = pd.read_csv("pairwise_dataset/dev_names.csv", encoding='utf-8', index_col=0)
test_names = pd.read_csv("pairwise_dataset/test_names.csv", encoding='utf-8', index_col=0)


out_file = open("entities/per_names.json", "w")
json.dump({'dev': {"in_train": dev_names['pos_name'].tolist(), "out_train": dev_names['neg_name'].tolist()},
           'test': {"in_train": test_names['pos_name'].tolist(), "out_train": test_names['neg_name'].tolist()},
           "declarative_prompt": de_prompts,
           "exclamatory_prompt": ex_prompts,
           "imperative_prompt": im_prompts,
           "interrogative_prompt": in_prompts,
           "empty_prompt": manual_prompts[0],
           "one_prompt": manual_prompts[1],
           "mix_prompt": manual_prompts[1:]
           }, 
           out_file, indent = 6, ensure_ascii=False)
out_file.close()