import numpy as np

with open("exclamatory_sentences.txt", encoding='utf-8') as f:
    prompts = f.read()
    # print(prompts)
prompts = prompts.split("\n")

for prompt in prompts:
    if prompts.count(prompt)> 1:
        print(prompt, prompts.count(prompt))

