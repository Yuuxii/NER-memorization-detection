# import torch
# import tensorflow
import argparse
import sys
from timeit import default_timer as timer
import torch
from utils import load_white_box_model, load_dataset, model_prediction, get_start_num_tokens
import os

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='base', help='choose from [albert-base, albert-large, bert-base, bert-large, roberta-base, roberta-large]')
    parser.add_argument('--prompt', default='empty', help='choose from [empty, one, mix, declarative, exclamatory, imperative, interrogative, all]')    
    # parser.add_argument('--use_data', default='dev', help=) 
    # parser.add_argument('--use_pre', action='store_true') 
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--entity_type', default='mix', help='choose from [per, loc, org]')
    args = parser.parse_args()
    print('args:', args)
    
    ## load black-box model
    # nlp = load_black_box_model(args)
    ## load white-box model
    ner_tokenizer, ner_model = load_white_box_model(args)
    ner_model = ner_model.to(device)
    print('model param: ', ner_model.num_parameters())
    
    ## load dataset
    dev_data, test_data, prompts = load_dataset(args)

    print("---"*10)
    start = timer()

    ## get model predictions and scores for dev data and test data
    # dev data
    dev_model_predictions = model_prediction(args, prompts, dev_data, ner_tokenizer, ner_model, 'dev')
    
    # test data
    test_model_predictions = model_prediction(args, prompts, test_data, ner_tokenizer, ner_model, 'test')

    ###### analysis ######
    ## rank the prompts based on the attack successful rate (asr)
    from analysis import rank_prompt_asr, get_single_multiple_word_analysis
    analysis_outs_path = 'analysis_outs/'+args.entity_type + '/'

    if not os.path.isdir(analysis_outs_path):
        # os.mkdir('analysis_outs/')
        os.mkdir(analysis_outs_path)

    ##  dev
    wht_dev_ranked_prompts_df = rank_prompt_asr(dev_model_predictions)
    wht_dev_ranked_prompts_df.to_csv( analysis_outs_path + args.model +'_'+ args.prompt +'_' + 'dev' +'.csv', encoding='utf-8')    
    get_start_num_tokens(args, wht_dev_ranked_prompts_df, analysis_outs_path, ner_tokenizer)
    
    dev_multi_word_analysis, dev_single_word_analysis = get_single_multiple_word_analysis(dev_model_predictions)
    dev_multi_word_analysis.to_csv( analysis_outs_path + "num_name_token_analysis/" + args.model +'_'+ args.prompt + '_multi_token_dev' +'.csv', encoding='utf-8')
    dev_single_word_analysis.to_csv( analysis_outs_path + "num_name_token_analysis/" + args.model +'_'+ args.prompt + '_single_token_dev' +'.csv', encoding='utf-8')

    ## based on the rank of the dev data, we can get the best three prompts and test on the test data
    wht_best_3_prompts = wht_dev_ranked_prompts_df['prompt'][:3].tolist()
    print('white box model best 3 prompts in dev data: ', wht_best_3_prompts)
    wht_worst_prompt = wht_dev_ranked_prompts_df['prompt'].tolist()[-1]

    ## test
    wht_test_ranked_prompts_df = rank_prompt_asr(test_model_predictions)  
    wht_test_ranked_prompts_df.to_csv(analysis_outs_path + args.model +'_'+ args.prompt +'_' + 'test' +'.csv', encoding='utf-8')   

    test_multi_word_analysis, test_single_word_analysis = get_single_multiple_word_analysis(test_model_predictions)
    test_multi_word_analysis.to_csv( analysis_outs_path + "num_name_token_analysis/" + args.model +'_'+ args.prompt + '_multi_token_test' +'.csv', encoding='utf-8')
    test_single_word_analysis.to_csv( analysis_outs_path + "num_name_token_analysis/" + args.model +'_'+ args.prompt + '_single_token_test' +'.csv', encoding='utf-8')

    
    wht_test_ranked_prompts_df=wht_test_ranked_prompts_df.reset_index()

    for i, prompt in enumerate(wht_best_3_prompts):
        index = wht_test_ranked_prompts_df.index[wht_test_ranked_prompts_df['prompt']==prompt].tolist()
        print('white box top {} prompt ({}) in dev get rank {} in test'.format(i, prompt, index))
    
    wht_worst_prompt_in_test = wht_test_ranked_prompts_df.index[wht_test_ranked_prompts_df['prompt']==wht_worst_prompt].tolist()
    print('white box model worst prompt ({}) in dev get rank {} in test'.format(wht_worst_prompt, wht_worst_prompt_in_test))
    
    time_needed = timer() - start
    print("Time needed for experiment:", time_needed)

