from datasets import load_dataset
import pandas as pd

dataset = load_dataset("conll2003")

train_tokens_list = dataset['train']['tokens']

all_sents = ""
for sent_tokens in train_tokens_list:
    for sent_token in sent_tokens:
        all_sents += sent_token
        all_sents += " "

dev_names = pd.read_csv("dev_names.csv", encoding='utf-8', index_col=0) 
test_names = pd.read_csv("test_names.csv", encoding='utf-8', index_col=0) 

dev_pos = dev_names['pos_name'].tolist()
dev_neg = dev_names['neg_name'].tolist()

test_pos = test_names['pos_name'].tolist()
test_neg = test_names['neg_name'].tolist()

for name in dev_pos:
    if name not in all_sents:
        print('this name from dev should be in train data: ', name)

for name in dev_neg:
    if name in all_sents:
        print('this name from dev should not be in train data: ', name)

for name in test_pos:
    if name not in all_sents:
        print('this name from test should be in train data: ', name)

for name in test_neg:
    if name in all_sents:
        print('this name from test should not be in train data: ', name)

all_names = dev_pos + dev_neg + test_pos + test_neg
for idx, i in enumerate(all_names):
    count = all_names.count(i)
    if count > 1:
        print(idx, i, count)


