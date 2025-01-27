from analysis import prompt_name_rank_table, rank_prompt_asr
import pandas as pd
from transformers import AutoTokenizer
def load_tokenizer(model_name):
    ############Loading BERT NER models#########
    if model_name == "albert-base":
        # ALBERT base v2
        tokenizer = AutoTokenizer.from_pretrained("ArBert/albert-base-v2-finetuned-ner")
    
    elif model_name == 'albert-large':
        tokenizer = AutoTokenizer.from_pretrained("Gladiator/albert-large-v2_ner_conll2003")

    # elif args.model == 'distil':
    #     tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
    #     model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")

    elif model_name == "bert-base":
        #BERT Base NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        
    elif model_name == "bert-large":
        #BERT Large NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")

    elif model_name == 'roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("dominiqueblok/roberta-base-finetuned-ner")
    
    elif model_name == 'roberta-large':
        # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        # model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        #Roberta Large NER
        # tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        # model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")

        tokenizer = AutoTokenizer.from_pretrained("Gladiator/roberta-large_ner_conll2003")


    elif model_name == 'deberta':
        # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        # model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        #Roberta Large NER
        tokenizer = AutoTokenizer.from_pretrained("Gladiator/microsoft-deberta-v3-large_ner_conll2003")
    
    elif model_name == 't5':
        #T5 model
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/t5-base-conll03-english")  
    return tokenizer

models = ['albert-base', 'albert-large', 'bert-base', 'bert-large', 'roberta-base', 'roberta-large']

splits = ['dev']

analysis_path = 'analysis_outs/'

all_data_posi = []
all_data_len = []
min_len = []
for model in models:
    single_model_results = model_predictions = pd.DataFrame(columns=['prompt', 'wht_asr', 'name_start_token_pos'])
    for split in splits:
        prompt_rank  =  pd.read_csv(analysis_path + 'wht_' + model + '_' + split + '.csv', encoding='utf-8', index_col=0)
        prompt_rank = prompt_rank.reset_index(drop=True)
        prompt_rank['name_start_token_pos'] = [0 for i in range(len(prompt_rank['prompt']))]
        prompt_rank['total_num_tokens'] = [0 for i in range(len(prompt_rank['prompt']))]
        prompt_rank['reletive_token_pos'] = [0 for i in range(len(prompt_rank['prompt']))]
        ner_tokenizer = load_tokenizer(model)

        for idx, prompt in enumerate(prompt_rank['prompt'].tolist()):
            prompt_tokens = ner_tokenizer(prompt, return_tensors="pt")
            mask_tokens = ner_tokenizer('MASK', return_tensors="pt")
            

            name_start_token_pos = [prompt_tokens.input_ids.tolist()[0].index(i) for i in mask_tokens.input_ids.tolist()[0][1:-1]][0]
            # print(prompt, prompt_tokens, mask_tokens, name_start_token_pos)

            prompt_rank['name_start_token_pos'][idx]= name_start_token_pos
            prompt_rank['total_num_tokens'][idx]= len(prompt_tokens.input_ids.tolist()[0])
            prompt_rank['reletive_token_pos'][idx]= len(prompt_tokens.input_ids.tolist()[0])/name_start_token_pos

        print( model, split)

        # prompt_rank_posi = prompt_rank[['name_start_token_pos', 'wht_asr']].copy()
        # print(prompt_rank_posi.corr(numeric_only=True, method="kendall"))

        # prompt_rank_len = prompt_rank[['total_num_tokens', 'wht_asr']].copy()
        # print(prompt_rank_len.corr(numeric_only=True, method="kendall"))

        # prompt_rank_len = prompt_rank[['reletive_token_pos', 'wht_asr']].copy()
        # print(prompt_rank_len.corr(numeric_only=True, method="kendall"))
        # prompt_rank.to_csv(analysis_path + 'wht_' + model + '_' + split + '_start_token.csv', encoding ='utf-8')    
        model_data_posi = [[] for i in prompt_rank['name_start_token_pos'].unique().tolist()]
        for num, asr in zip(prompt_rank['name_start_token_pos'], prompt_rank['wht_asr']):
            model_data_posi[num-1].append(asr) 

        all_data_posi.append(model_data_posi)    

        model_data_len = [[] for i in prompt_rank['total_num_tokens'].unique().tolist()]
        min_value = min(prompt_rank['total_num_tokens'].unique().tolist())
        for num, asr in zip(prompt_rank['total_num_tokens'], prompt_rank['wht_asr']):
            model_data_len[num-min_value].append(asr) 

        all_data_len.append(model_data_len)    
        min_len.append(min_value)

# plot
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# fig, axs = plt.subplots(1, 6, figsize=(20, 4))


# axs[0].set_title('ALBERT-B',fontweight="bold", size=15)
# axs[1].set_title("ALBERT-L",fontweight="bold", size=15)
# axs[2].set_title('BERT-B',fontweight="bold", size=15)
# axs[3].set_title("BERT-L",fontweight="bold", size=15)
# axs[4].set_title('RoBERTa-B',fontweight="bold", size=15)
# axs[5].set_title("RoBERTa-L", fontweight="bold", size=15)

# for i in range(6):
#     axs[i].boxplot(all_data_posi[i], 0, '')
#     axs[i].xaxis.set_tick_params(labelsize=14)
#     axs[i].yaxis.set_tick_params(labelsize=14)
#     axs[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# plt.savefig('plots/all_models_posi.jpg')



fig, axs = plt.subplots(1, 6, figsize=(20, 4))

axs[0].set_title('ALBERT-B',fontweight="bold", size=15)
axs[1].set_title("ALBERT-L",fontweight="bold", size=15)
axs[2].set_title('BERT-B',fontweight="bold", size=15)
axs[3].set_title("BERT-L",fontweight="bold", size=15)
axs[4].set_title('RoBERTa-B',fontweight="bold", size=15)
axs[5].set_title("RoBERTa-L", fontweight="bold", size=15)
axs[1].set_title("ALBERT-L",fontweight="bold", size=15)
axs[2].set_title('BERT-B',fontweight="bold", size=15)
axs[3].set_title("BERT-L",fontweight="bold", size=15)
axs[4].set_title('RoBERTa-B',fontweight="bold", size=15)
axs[5].set_title("RoBERTa-L", fontweight="bold", size=15)

for i in range(6):
    axs[i].boxplot(all_data_len[i], 0, '')
    axs[i].set_xticks(axs[i].get_xticks()[::2])
    axs[i].set_xticklabels([min_len[i]+m for m in range(0, len(all_data_len[i]), 2)])
    axs[i].xaxis.set_tick_params(labelsize=14)
    axs[i].yaxis.set_tick_params(labelsize=14)
    axs[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

plt.savefig('plots/all_models_len.jpg')

