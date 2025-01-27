# import torch
# import tensorflow

import argparse
 
import sys
from timeit import default_timer as timer
import torch
from utils import load_white_box_model, load_dataset, model_prediction
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='base', help='choose from [albert, distil, base, large, roberta]')
    parser.add_argument('--prompt', default=None)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    print('args:', args)

    ## load black box model
    # nlp = load_black_box_model(args)
    ## load white box model
    ner_tokenizer, ner_model = load_white_box_model(args)
    ner_model = ner_model.to(device)
    ## load pairwise dataset
    dev_data, test_data = load_dataset()

    ## load prompts
    prompts = [args.prompt]

    print("---"*10)
    start = timer()

    ## get model predictions and scores for dev data and test data
    dev_model_predictions = model_prediction(args, prompts, dev_data, ner_tokenizer, ner_model, 'dev')
    
    test_model_predictions = model_prediction(args, prompts, test_data, ner_tokenizer, ner_model, 'test')

    ## analysis
    from analysis import rank_prompt_asr
    ##  dev
    wht_dev_ranked_prompts_df = rank_prompt_asr(dev_model_predictions)
    # dev_ranked_prompts_df.to_csv('analysis_outs/' + args.model +'_'+ args.prompt_group +'_' + 'dev' +'.csv', encoding='utf-8')    
    print(wht_dev_ranked_prompts_df)

    ## test
    wht_test_ranked_prompts_df = rank_prompt_asr(test_model_predictions)
    # test_ranked_prompts_df.to_csv('analysis_outs/' + args.model +'_'+ args.prompt_group +'_' + 'test' +'.csv', encoding='utf-8')    
    print(wht_test_ranked_prompts_df)

    time_needed = timer() - start
    print("Time needed for experiment:", time_needed)
    sys.stdout.close()
