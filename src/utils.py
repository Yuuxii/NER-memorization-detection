from cmd import PROMPT
from torch._C import device
from torch.cuda import device_count
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from transformers import pipeline
import json
import numpy as np
import pandas as pd
import os
from transformers.pipelines.image_segmentation import Prediction
import torch
import random

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

def get_white_box_score(args, name_sentence, name, tokenizer, model):
    # print(name_sentence)
    # print(name)
    sent_tokens = tokenizer(name_sentence, return_tensors="pt").to(device)
    name_tokens = tokenizer(name, return_tensors="pt")
    # print(sent_tokens.input_ids.tolist()[0])
    # print(name_tokens.input_ids.tolist()[0][1:-1])
    
    if name_tokens.input_ids.tolist()[0][-2] not in sent_tokens.input_ids.tolist()[0]:
        in_train_names = [sent_tokens.input_ids.tolist()[0].index(i) for i in name_tokens.input_ids.tolist()[0][1:-2]]
    else:
        in_train_names = [sent_tokens.input_ids.tolist()[0].index(i) for i in name_tokens.input_ids.tolist()[0][1:-1]]
        
    model.eval()
    # print(sent_tokens, name_tokens, in_train_names)
    with torch.no_grad():
        output = model(**sent_tokens)
    # print(model)
    logits = output.logits
    # print(output)

    probs = torch.nn.Softmax(2)(logits)

    if args.entity_type == 'per':
        
        # print(probs.max(2))
        # mean value of the confidence across all name tokens
        if args.model in ['bert-base', 'bert-large']:
            name_prob = np.mean([max(probs[0][i][3].item(), probs[0][i][4].item()) for i in in_train_names])
        
        elif args.model in ['albert-base', 'albert-large', 'roberta-base', 'roberta-large']:
            # print(probs.max(2))
            name_prob = np.mean([max(probs[0][i][1].item(), probs[0][i][2].item()) for i in in_train_names])
    
    elif args.entity_type == 'loc':
        if args.model in ['bert-base', 'bert-large']:
            
            name_prob = np.mean([max(probs[0][i][7].item(), probs[0][i][8].item()) for i in in_train_names])
        
        elif args.model in ['albert-base', 'albert-large', 'roberta-base', 'roberta-large']:
            
            name_prob = np.mean([max(probs[0][i][5].item(), probs[0][i][6].item()) for i in in_train_names])
    
    elif args.entity_type == 'org':
        if args.model in ['bert-base', 'bert-large']:
         
            name_prob = np.mean([max(probs[0][i][5].item(), probs[0][i][6].item()) for i in in_train_names])
        
        elif args.model in ['albert-base', 'albert-large', 'roberta-base', 'roberta-large']:
           
            name_prob = np.mean([max(probs[0][i][3].item(), probs[0][i][4].item()) for i in in_train_names])
    else:   
        print("unknown entity types")

    pred_name = [sent_tokens.input_ids[0][i] for i in in_train_names]
  
    return name_prob, tokenizer.decode(pred_name)

def load_white_box_model(args):
    ############Loading BERT NER models#########
    if args.model == "albert-base":
        # ALBERT base v2
        tokenizer = AutoTokenizer.from_pretrained("ArBert/albert-base-v2-finetuned-ner")
        model = AutoModelForTokenClassification.from_pretrained("ArBert/albert-base-v2-finetuned-ner")
    
    elif args.model == 'albert-large':
        tokenizer = AutoTokenizer.from_pretrained("Gladiator/albert-large-v2_ner_conll2003")
        model = AutoModelForTokenClassification.from_pretrained("Gladiator/albert-large-v2_ner_conll2003")

    elif args.model == "bert-base":
        #BERT Base NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        
    elif args.model == "bert-large":
        #BERT Large NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    elif args.model == 'roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("dominiqueblok/roberta-base-finetuned-ner")
        model = AutoModelForTokenClassification.from_pretrained("dominiqueblok/roberta-base-finetuned-ner")
    
    elif args.model == 'roberta-large':
        tokenizer = AutoTokenizer.from_pretrained("Gladiator/roberta-large_ner_conll2003")
        model = AutoModelForTokenClassification.from_pretrained("Gladiator/roberta-large_ner_conll2003")


    elif args.model == 'deberta':
        # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        # model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        #Roberta Large NER
        tokenizer = AutoTokenizer.from_pretrained("Gladiator/microsoft-deberta-v3-large_ner_conll2003")
        model = AutoModelForTokenClassification.from_pretrained("Gladiator/microsoft-deberta-v3-large_ner_conll2003")
    
    elif args.model == 't5':
        #T5 model
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/t5-base-conll03-english")  
        model = AutoModelForSeq2SeqLM.from_pretrained("dbmdz/t5-base-conll03-english")

    return tokenizer, model


def load_dataset(args):

    data = json.load(open('entities/%s_names.json'%args.entity_type, 'r'))
 
 
    dev_names = data['dev']
    test_names = data['test']

    if args.prompt in ["empty", "one", "mix", "declarative", "exclamatory", "imperative", "interrogative"]:
        prompts = data['%s_prompt'%args.prompt]
    elif args.prompt == "all":
        prompts = data['declarative_prompt'] + data['exclamatory_prompt'] +data['imperative_prompt'] +data['interrogative_prompt'] 
    else:
        print("unrecognized prompt type")
 
    return dev_names, test_names, prompts


def model_prediction(args, prompts, data, ner_tokenizer, ner_model, data_type):
    # print(prompts)
    prediction_results_path = "model_predictions/" + args.entity_type + '/'
    if not os.path.isdir(prediction_results_path):
        os.mkdir(prediction_results_path)

    model_prediction_file =  prediction_results_path + args.model +'_'+ args.prompt +'_' + data_type + ".csv"
    
    if os.path.isfile(model_prediction_file):
        model_predictions = pd.read_csv(model_prediction_file, encoding='utf-8', index_col=0)

        for prompt in prompts:


            if prompt not in model_predictions['prompt'].tolist():

                for idx, (in_train_name, out_train_name) in enumerate(zip(data["in_train"], data["out_train"])):   
                    #negative sample
                    sentence_neg = prompt.replace("MASK", out_train_name)
                    # white-box setting
                    wht_score_neg, wht_pred_out_train_name = get_white_box_score(args, sentence_neg, out_train_name, ner_tokenizer, ner_model)
                    dict_neg = {'prompt' : prompt, 
                                'is_in_train': 0, 
                                'name': out_train_name, 
                                'model_pred' : wht_pred_out_train_name,
                                'confidence' : wht_score_neg
                                }
                    model_predictions = pd.concat([model_predictions, pd.DataFrame([dict_neg])], ignore_index=True)
                    
                    #positive sample
                    sentence_pos = prompt.replace("MASK", in_train_name)
                    # white-box setting
                    wht_score_pos, wht_pred_in_train_name = get_white_box_score(args, sentence_pos, in_train_name, ner_tokenizer, ner_model)
                    
                    dict_pos = {'prompt' : prompt, 
                                'is_in_train': 1, 
                                'name': in_train_name, 
                                'model_pred' : wht_pred_in_train_name,
                                'confidence' : wht_score_pos
                                }
                    model_predictions = pd.concat([model_predictions, pd.DataFrame([dict_pos])], ignore_index=True)

            #save results
        
        model_predictions.to_csv(model_prediction_file, encoding ='utf-8')
        
    else:
        print("Model Predictions started...")
        model_predictions = pd.DataFrame(columns=['prompt', 'is_in_train', 'name', 'model_pred', 'confidence'])
        if args.prompt == 'mix':
            print('mix prompt')
            for idx, (in_train_name, out_train_name) in enumerate(zip(data["in_train"], data["out_train"])):   
                #negative sample
                prompt = random.choice(prompts)
                sentence_neg = prompt.replace("MASK", out_train_name)
            
                wht_score_neg, wht_pred_out_train_name = get_white_box_score(args, sentence_neg, out_train_name, ner_tokenizer, ner_model)
                dict_neg = {'prompt' : 'mix', 
                            'is_in_train': 0, 
                            'name': out_train_name, 
                            'model_pred' : wht_pred_out_train_name,
                            'confidence' : wht_score_neg
                            }
                model_predictions = pd.concat([model_predictions, pd.DataFrame([dict_neg])], ignore_index=True)
                
                #positive sample
                prompt = random.choice(prompts)
                sentence_pos = prompt.replace("MASK", in_train_name)
                
                wht_score_pos, wht_pred_in_train_name = get_white_box_score(args, sentence_pos, in_train_name, ner_tokenizer, ner_model)
                
                dict_pos = {'prompt' : 'mix',  
                            'is_in_train': 1, 
                            'name': in_train_name, 
                            'model_pred' : wht_pred_in_train_name,
                            'confidence' : wht_score_pos
                            }
                model_predictions = pd.concat([model_predictions, pd.DataFrame([dict_pos])], ignore_index=True)
        else:
            for prompt in prompts:
                for idx, (in_train_name, out_train_name) in enumerate(zip(data["in_train"], data["out_train"])):   
                    #negative sample
                    sentence_neg = prompt.replace("MASK", out_train_name)
                    # white-box setting
                    wht_score_neg, wht_pred_out_train_name = get_white_box_score(args, sentence_neg, out_train_name, ner_tokenizer, ner_model)
                    dict_neg = {'prompt' : prompt, 
                                'is_in_train': 0, 
                                'name': out_train_name, 
                                'model_pred' : wht_pred_out_train_name,
                                'confidence' : wht_score_neg
                                }
                    model_predictions = pd.concat([model_predictions, pd.DataFrame([dict_neg])], ignore_index=True)
                    
                    #positive sample
                    sentence_pos = prompt.replace("MASK", in_train_name)
                    # white-box setting
                    wht_score_pos, wht_pred_in_train_name = get_white_box_score(args, sentence_pos, in_train_name, ner_tokenizer, ner_model)
                    
                    dict_pos = {'prompt' : prompt, 
                                'is_in_train': 1, 
                                'name': in_train_name, 
                                'model_pred' : wht_pred_in_train_name,
                                'confidence' : wht_score_pos
                                }
                    model_predictions = pd.concat([model_predictions, pd.DataFrame([dict_pos])], ignore_index=True)

        #save results
    
        model_predictions.to_csv(model_prediction_file, encoding ='utf-8')
    
    print('predictions save to: ', prediction_results_path)

    return model_predictions 

def get_start_num_tokens(args, analysis_df, analysis_outs_path, tokenizer):

    names_start_positions = []
    prompt_lens = []
    for prompt in analysis_df['prompt'].tolist():
        sent_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        name_tokens = tokenizer('MASK', return_tensors="pt")

        names_start_positions.append(sent_tokens.input_ids.tolist()[0].index(name_tokens.input_ids.tolist()[0][1]) + 1)
        prompt_lens.append(sent_tokens.input_ids.shape[1]) 
    
    analysis_df['name_start_token_pos'] = names_start_positions
    analysis_df['total_num_tokens'] = prompt_lens

    analysis_df.to_csv(analysis_outs_path + args.model +'_'+ args.prompt +'_' + 'dev_start_token' +'.csv', encoding='utf-8') 


