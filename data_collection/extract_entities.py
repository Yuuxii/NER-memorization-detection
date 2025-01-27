from datasets import load_dataset
import json
from random import sample, shuffle

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString) 


idx_label = {'O': 0, 
             'B-PER': 1, 
             'I-PER': 2, 
             'B-ORG': 3, 
             'I-ORG': 4, 
             'B-LOC': 5, 
             'I-LOC': 6, 
             'B-MISC': 7, 
             'I-MISC': 8}

in_train_dataset = load_dataset("conll2003")
out_train_dataset = load_dataset("unimelb-nlp/wikiann", "en")

train_tokens_list = in_train_dataset['train']['tokens']
train_ners_list = in_train_dataset['train']['ner_tags']

out_train_tokens_list = out_train_dataset['train']['tokens']
out_train_tokens_list.extend(out_train_dataset['validation']['tokens'])
out_train_tokens_list.extend(out_train_dataset['test']['tokens'])

out_train_ners_list = out_train_dataset['train']['ner_tags']
out_train_ners_list.extend(out_train_dataset['validation']['ner_tags'])
out_train_ners_list.extend(out_train_dataset['test']['ner_tags'])

## in_train entities
per_names = []
org_names = []
loc_names = []
all_ners = []
for sent_tokens, sent_ners in zip(train_tokens_list, train_ners_list):
   

    for idx, (sent_token, sent_ner) in enumerate(zip(sent_tokens, sent_ners)):
        if sent_ner == 1 and sent_token[0].isupper():
            per = sent_token
            for ner, token in zip(sent_ners[idx+1 :], sent_tokens[idx+1 :]):
                if ner != 2:
                    if per not in per_names and len(per.split(" ")) > 1:
                        per_names.append(per)
                    break
                else:
                    per+= " " + token

        elif sent_ner == 3 and sent_token[0].isupper():
            org = sent_token
            # print(sent_ner)
            for ner, token in zip(sent_ners[idx+1 :], sent_tokens[idx+1 :]):
                if ner != 4:
                    if org not in org_names:
                        org_names.append(org)
                    break
                else:
                    org+= " " + token

        elif sent_ner == 5 and sent_token[0].isupper():
            loc = sent_token
            for ner, token in zip(sent_ners[idx+1 :], sent_tokens[idx+1 :]):
                if ner != 6:
                    if loc not in loc_names:
                        loc_names.append(loc)
                    break
                else:
                    loc+= " " + token



## out_train_entities
# out_per_names = []
out_org_names = []
out_loc_names = []
out_all_ners = []
for sent_tokens, sent_ners in zip(out_train_tokens_list, out_train_ners_list):
   

    for idx, (sent_token, sent_ner) in enumerate(zip(sent_tokens, sent_ners)):
        # if sent_ner == 1:
        #     per = sent_token
        #     for ner, token in zip(sent_ners[idx+1 :], sent_tokens[idx+1 :]):
        #         if ner != 2:
        #             if per not in per_names and len(per.split(" ")) > 1:
        #                 out_per_names.append(per)
        #             break
        #         else:
        #             per+= " " + token

        if sent_ner == 3 and sent_token[0].isupper() and not has_numbers(sent_token):
            org = sent_token
            # print(sent_ner)
            for ner, token in zip(sent_ners[idx+1 :], sent_tokens[idx+1 :]):
                if ner != 4:
                    if org not in org_names and org not in out_org_names:
                        out_org_names.append(org)
                    break
                elif not has_numbers(token):
                    org += " " + token

        elif sent_ner == 5 and sent_token[0].isupper() and not has_numbers(sent_token):
            loc = sent_token
            for ner, token in zip(sent_ners[idx+1 :], sent_tokens[idx+1 :]):
                if ner != 6:
                    if loc not in loc_names and loc not in out_loc_names:
                        out_loc_names.append(loc)
                    break
                elif not has_numbers(token):
                    loc+= " " + token


print(len(out_org_names))
print(len(org_names))

print(len(out_loc_names))
print(len(loc_names))
out_org_names = sample(out_org_names, len(org_names))
out_loc_names = sample(out_loc_names, len(loc_names))
shuffle(org_names)
shuffle(loc_names) 

print(len(org_names[: round(len(org_names)/2)])) 
print(len(out_org_names[: round(len(org_names)/2)]))
print(len(org_names[round(len(org_names)/2) :]))
print(len(out_org_names[round(len(org_names)/2) :]))

print(len(loc_names[: round(len(loc_names)/2)]))
print(len(out_loc_names[: round(len(loc_names)/2)]))
print(len(loc_names[round(len(loc_names)/2) :]))
print(len(out_loc_names[round(len(loc_names)/2) :])) 
           


loc_de_prompts = []
loc_ex_prompts = []
loc_im_prompts = []
loc_in_prompts = []
loc_manual_prompts = ["MASK", "I am at MASK.", "I like MASK.", "MASK is a good place.", "Meet at MASK.", "Are you live in MASK?"]

org_de_prompts = []
org_ex_prompts = []
org_im_prompts = []
org_in_prompts = []
org_manual_prompts = ["MASK", "I work for MASK.", "I like MASK.", "MASK is a good organization.", "See you in MASK.", "Do you know MASK?"]

# print('Use {} prompt group'.format(args.prompt))
with open("entities/loc_declarative_exclamatory_imperative_interrogative.txt", encoding='utf-8') as f:
    prompts = f.read()
    prompts = prompts.split("\n")

    loc_de_prompts = prompts[:100]
    loc_ex_prompts = prompts[100:200]
    loc_im_prompts = prompts[200:300]
    loc_in_prompts = prompts[300:400]

with open("entities/org_declarative_exclamatory_imperative_interrogative.txt", encoding='utf-8') as f:
    prompts = f.read()
    prompts = prompts.split("\n")

    org_de_prompts = prompts[:100]
    org_ex_prompts = prompts[100:200]
    org_im_prompts = prompts[200:300]
    org_in_prompts = prompts[300:400]



# the json file where the output must be stored
# out_file = open("entities/train_per_names.json", "w", encoding='utf-8')
# json.dump({'in_train_per_names': per_names}, out_file, indent = 6, ensure_ascii=False)
# out_file.close()

out_org_names = sample(out_org_names, len(org_names))
out_loc_names = sample(out_loc_names, len(loc_names))
shuffle(org_names)
shuffle(loc_names) 



out_file = open("entities/org_names_test.json", "w")
json.dump({'dev': {"in_train": org_names[: round(len(org_names)/2)], "out_train": out_org_names[: round(len(org_names)/2)]},
           'test': {"in_train": org_names[round(len(org_names)/2) :], "out_train": out_org_names[round(len(org_names)/2) :]},
           "declarative_prompt": org_de_prompts,
           "exclamatory_prompt": org_ex_prompts,
           "imperative_prompt": org_im_prompts,
           "interrogative_prompt": org_in_prompts,
           "empty_prompt": org_manual_prompts[0],
           "one_prompt": org_manual_prompts[1],
           "mix_prompt": org_manual_prompts[1:]
           }, 
           out_file, indent = 6, ensure_ascii=False)
out_file.close()

out_file = open("entities/loc_names_test.json", "w")
json.dump({'dev': {"in_train": loc_names[: round(len(loc_names)/2)], "out_train": out_loc_names[: round(len(loc_names)/2)]},
           'test': {"in_train": loc_names[round(len(loc_names)/2) :], "out_train": out_loc_names[round(len(loc_names)/2) :]},
           "declarative_prompt": loc_de_prompts,
           "exclamatory_prompt": loc_ex_prompts,
           "imperative_prompt": loc_im_prompts,
           "interrogative_prompt": loc_in_prompts,
           "empty_prompt": loc_manual_prompts[0],
           "one_prompt": loc_manual_prompts[1],
           "mix_prompt": loc_manual_prompts[1:]
           }, 
           out_file, indent = 6, ensure_ascii=False)
out_file.close()