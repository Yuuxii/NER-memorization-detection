# import torch
# import tensorflow
import argparse
import sys
from timeit import default_timer as timer
import torch
import os
from utils import load_white_box_model, load_dataset, model_prediction
import pandas as pd
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='base', help='choose from [albert-base, albert-large, bert-base, bert-large, roberta-base, roberta-large]')
    # parser.add_argument('--prompt', default='simply', help='choose from [simply, declarative, interrogative, general]')    
    # parser.add_argument('--use_data', default='dev', help=) 
    # parser.add_argument('--use_pre', action='store_true') 
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    print('args:', args)
    
    ## load black-box model
    # nlp = load_black_box_model(args)
    ## load white-box model
    # ner_tokenizer, ner_model = load_white_box_model(args)
    # ner_model = ner_model.to(device)
    # print('model param: ', ner_model.num_parameters())
    ## load pairwise dataset
    # dev_data, test_data = load_dataset()

    ## load prompts
    # prompts = load_prompts(args)

    print("---"*10)
    start = timer()

    ## get model predictions and scores for dev data and test data
    prediction_results_path = "model_predictions/org/"
    
    # dev data
    dev_model_predictions = pd.DataFrame(columns=['prompt', 'is_in_train', 'name', 'model_pred', 'wht_confidence'])
    for prompt_type in ['declarative', 'exclamatory', 'imperative', 'interrogative']:
        type_model_prediction = pd.read_csv(prediction_results_path + args.model +'_'+ prompt_type +'_dev.csv', encoding='utf-8', index_col=0)
        dev_model_predictions = pd.concat([dev_model_predictions, type_model_prediction], ignore_index=True) 
    
    # test data
    test_model_predictions = pd.DataFrame(columns=['prompt', 'is_in_train', 'name', 'model_pred', 'wht_confidence'])
    for prompt_type in ['declarative', 'exclamatory', 'imperative', 'interrogative']:
        type_model_prediction = pd.read_csv(prediction_results_path + args.model +'_'+ prompt_type +'_test.csv', encoding='utf-8', index_col=0)
        test_model_predictions = pd.concat([test_model_predictions, type_model_prediction], ignore_index=True) 
   
    ###### ensemble ######
    from analysis import prompt_name_rank_table, rank_asr_on_scores
    analysis_outs_path = 'analysis_outs/org/'
    ##  load analysis
    dev_analysis = pd.DataFrame(columns=['prompt', 'asr'])
    for prompt_type in ['declarative', 'exclamatory', 'imperative', 'interrogative']:
        dev_ranked_prompts_df = pd.read_csv(analysis_outs_path+  args.model +'_' + prompt_type + '_dev' +'.csv', encoding='utf-8', index_col=0)    
        dev_analysis = pd.concat([dev_analysis, dev_ranked_prompts_df], ignore_index=True) 

    dev_prompt_name_rank, dev_prompt_name_score, dev_prompt_name_score_result = prompt_name_rank_table(dev_model_predictions, dev_analysis)
    test_prompt_name_rank, test_prompt_name_score, test_prompt_name_score_result = prompt_name_rank_table(test_model_predictions, dev_analysis)
    
    ## save ensemble results
    ensemble_outs_path = 'ensemble_outs/org/'
    if not os.path.isdir(ensemble_outs_path):
        os.mkdir(ensemble_outs_path)

    dev_prompt_name_rank.to_csv(ensemble_outs_path + args.model + '_dev_rank.csv', encoding='utf-8')
    dev_prompt_name_score.to_csv(ensemble_outs_path + args.model  +'_dev_score.csv', encoding='utf-8')
    # dev_prompt_name_rank_result.to_csv(ensemble_outs_path + args.model + '_dev_rank_result.csv', encoding='utf-8')
    dev_prompt_name_score_result.to_csv(ensemble_outs_path + args.model + '_dev_full_score_result.csv', encoding='utf-8')
    rank_asr_on_scores(dev_prompt_name_score_result)
    # rank_asr_on_rank(dev_prompt_name_rank_result)

    test_prompt_name_rank.to_csv(ensemble_outs_path + args.model + '_test_rank.csv', encoding='utf-8')
    test_prompt_name_score.to_csv(ensemble_outs_path + args.model  +'_test_score.csv', encoding='utf-8')
    # test_prompt_name_rank_result.to_csv(ensemble_outs_path + args.model + '_test_rank_result.csv', encoding='utf-8')
    test_prompt_name_score_result.to_csv(ensemble_outs_path + args.model + '_test_full_score_result.csv', encoding='utf-8')

    rank_asr_on_scores(test_prompt_name_score_result)
    # rank_asr_on_rank(test_prompt_name_rank_result)

    time_needed = timer() - start
    print("Time needed for experiment:", time_needed)

