from multiprocessing.pool import AsyncResult
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())


def rank_prompt_asr(df):
    
    wht_winner_dist = pd.DataFrame(columns=['prompt', 'asr'])       
    for prompt in df.prompt.unique():

        # white box rank
        wht_prompt_temp = df.loc[(df['prompt'] == prompt)]

        wht_prompt_temp = wht_prompt_temp.sort_values(by=['confidence'], ascending=False)   
     
        wht_prompt_is_in_train_col_rank = wht_prompt_temp['is_in_train'].tolist()
        # print('wht_prompt_is_in_train_col_rank: ', wht_prompt_is_in_train_col_rank)
        wht_winners = 0
        
        for i in range(len(wht_prompt_is_in_train_col_rank)):
            # print('i:', i)

            # any name beyond a pos name is counted as a winner case for this prompt
            if wht_prompt_is_in_train_col_rank[i] == 1:
                wht_winners += wht_prompt_is_in_train_col_rank[i+1:].count(0)
        ## in total, we have total_num_name*total_num_name/4 pairs
        # print(wht_winners)
        wht_asr = wht_winners/(len(wht_prompt_is_in_train_col_rank)*len(wht_prompt_is_in_train_col_rank)/4)
        # print('wht_asr:', wht_asr)
        wht_temp_dict = {'prompt': prompt, 'asr': round(wht_asr*100,2)}
        wht_winner_dist = pd.concat([wht_winner_dist, pd.DataFrame([wht_temp_dict])], ignore_index=True)
        
    wht_ranked_prompts_df = wht_winner_dist.sort_values(by=['asr'], ascending=False)
    
    return wht_ranked_prompts_df

def get_single_multiple_word_analysis(prediction_df):
    
    multi_word_prediction_df = prediction_df[prediction_df['name'].str.find(" ") > 0]
    single_word_prediction_df = prediction_df[prediction_df['name'].str.find(" ") == -1]
    print('(single_word_prediction_df): ', single_word_prediction_df['name'])
    print('multi_word_prediction_df:  ', multi_word_prediction_df['name'])
    multi_word_analysis = rank_prompt_asr(multi_word_prediction_df)
    single_word_analysis = rank_prompt_asr(single_word_prediction_df)

    return multi_word_analysis, single_word_analysis

def rank_asr_on_rank(df):

    gts = df['is_in_train'].values.tolist()
    mvs = df['mv'].values.tolist()
    votes = df['votes'].values.tolist()
    votes_in_train = votes[:int(len(votes)/2)]
    votes_not_in_train = votes[int(len(votes)/2):]
    right = 0
    best = 0
    for gt, mv, vote in zip(gts, mvs, votes):
        if gt == mv:
            right +=1
        if vote == 400 and gt != mv:
            best +=1
    mv_win = 0
    for in_train in votes_in_train:
        for not_in_train in votes_not_in_train:
            if in_train > not_in_train:
                mv_win +=1


    print('mv acc: ', right/len(gts))
    print('mv_win: ', mv_win/len(votes_in_train)/len(votes_not_in_train))
    print('oracle: ', 1 - best/len(gts))


def rank_asr_on_scores(df):

    df_avg_score = np.array(df.sort_values(by=['avg_score'], ascending=False).values.tolist()).T[-1].tolist()
    df_max_score = np.array(df.sort_values(by=['max_score'], ascending=False).values.tolist()).T[-1].tolist()
    df_min_score = np.array(df.sort_values(by=['min_score'], ascending=False).values.tolist()).T[-1].tolist()
    # df_min_5_score = np.array(df.sort_values(by=['min5_score'], ascending=False).values.tolist()).T[-1].tolist()
    # df_min_3_score = np.array(df.sort_values(by=['min3_score'], ascending=False).values.tolist()).T[-1].tolist()
    # df_top50_score = np.array(df.sort_values(by=['max50_score'], ascending=False).values.tolist()).T[-1].tolist()
    # df_top10_score = np.array(df.sort_values(by=['max10_score'], ascending=False).values.tolist()).T[-1].tolist()
    # df_top_5_score = np.array(df.sort_values(by=['max5_score'], ascending=False).values.tolist()).T[-1].tolist()
    # df_top_3_score = np.array(df.sort_values(by=['max3_score'], ascending=False).values.tolist()).T[-1].tolist()
    df_weighted_score = np.array(df.sort_values(by=['weighted_score'], ascending=False).values.tolist()).T[-1].tolist()

    # print(np.array(df.sort_values(by=['max_score'], ascending=False).values.tolist()).T[3].tolist())
    # prompt_rank = np.array(ranked_prompt_value).T[2].tolist()
    # 
    # for prompt_rank in [ df_avg_score, df_max_score,df_top_3_score, df_top_5_score, df_top10_score, df_top50_score, df_min_score, df_min_3_score, df_min_5_score, df_weighted_score]:
    for prompt_rank in [ df_avg_score, df_weighted_score, df_max_score, df_min_score]:
        winners = 0
        for i in range(len(prompt_rank)):
            # print(prompt_rank)
            # print(prompt_rank[i+1:].count(str('0')))
            if prompt_rank[i] == 1:
                winners += prompt_rank[i+1:].count(0)
            
        
        asr = winners/(len(prompt_rank)*len(prompt_rank)/4)
        print(round(asr*100,2))



def prompt_name_rank_table(df, acc_df=None):


    prompt_name_rank = pd.DataFrame() 
    prompt_name_score = pd.DataFrame() 
    
    # num_name = len(df['name'].drop_duplicates().tolist())
    # print(df['name'].unique().tolist())
    # name_list = df['name'].unique().tolist()
    # print('num_name: ', num_name, name_list)
    # pos_neg_name = []
    for prompt in acc_df.prompt.unique():
        df_prompt_temp = df.loc[(df['prompt'] == prompt)].drop_duplicates()
        # print(df_prompt_temp['confidence'].values.tolist())
        # for indx, i in enumerate(df_prompt_temp['confidence'].values.tolist()):
        #     if isinstance(i, str):
        #         print(indx, i)
        # df_prompt_temp['confidence'] = df_prompt_temp['confidence'].values.astype(float)
        try:
            prompt_rank_sorted = df_prompt_temp.sort_values(by=['confidence'], ascending=False)
        except:
            prompt_rank_sorted = df_prompt_temp.sort_values(by=['wht_confidence'], ascending=False)
        prompt_rank = prompt_rank_sorted['name'].tolist()
  
        df_prompt_pos = df_prompt_temp.loc[df_prompt_temp['is_in_train']==1]
        df_prompt_neg = df_prompt_temp.loc[df_prompt_temp['is_in_train']==0]
        
        num_pos_name = len(df_prompt_pos['name'].unique().tolist())
        num_neg_name = len(df_prompt_neg['name'].unique().tolist())

        # pos_neg_name.extend(df_prompt_pos['name'].unique().tolist())
        # pos_neg_name.extend(df_prompt_neg['name'].unique().tolist())
        try:
            prompt_name_score_pos = df_prompt_pos['confidence'].tolist()
            prompt_name_score_neg = df_prompt_neg['confidence'].tolist()
        except:
            prompt_name_score_pos = df_prompt_pos['wht_confidence'].tolist()
            prompt_name_score_neg = df_prompt_neg['wht_confidence'].tolist()
        prompt_name_rank_pos = [prompt_rank.index(df_prompt_pos['name'][i]) for i in df_prompt_pos.index]
        prompt_name_rank_neg = [prompt_rank.index(df_prompt_neg['name'][i]) for i in df_prompt_neg.index]

        # prompt_name_rank['name'] = df_prompt_temp['name'].unique().tolist()
        # prompt_name_score['name'] = df_prompt_temp['name'].unique().tolist()
        
        prompt_name_rank['name'] = df_prompt_pos['name'].unique().tolist() + df_prompt_neg['name'].unique().tolist()

        prompt_name_rank = pd.concat((prompt_name_rank, pd.DataFrame({prompt:np.array(prompt_name_rank_pos + prompt_name_rank_neg).astype(int)})), axis=1)
        prompt_name_score = pd.concat((prompt_name_score, pd.DataFrame({prompt:np.array(prompt_name_score_pos + prompt_name_score_neg).astype(float)})), axis=1)
 
    # print(prompt_name_score.T.iloc[1:].mean())
    # prompt_name_score = prompt_name_score.astype(float)
    # prompt_name_rank = prompt_name_rank.astype(int)
    prompt_name_rank_result = pd.DataFrame() 
    prompt_name_score_result = pd.DataFrame() 
    # a = [n if n not in pos_neg_name else None for n in name_list]
    # for i in a:
    #     if i is not None:
    #         print(i)
    
    # prompt_name_score_result['name'] = [df_prompt_pos['name'][i] for i in df_prompt_pos.index].append(df_prompt_neg['name'][i] for i in df_prompt_neg.index)
    prompt_name_score_result['avg_score'] = prompt_name_score.T.iloc[1:].mean().tolist()
    prompt_name_score_result['max_score'] = prompt_name_score.T.iloc[1:].max().tolist()
    prompt_name_score_result['min_score'] = prompt_name_score.T.iloc[1:].min().tolist()
    # prompt_name_score_result['max50_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    # prompt_name_score_result['max10_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    # prompt_name_score_result['max5_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    # prompt_name_score_result['max3_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    # prompt_name_score_result['min5_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    # prompt_name_score_result['min3_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    prompt_name_score_result['weighted_score'] = [0.0 for i in range(num_pos_name+num_neg_name)]
    

    # print('acc: ', acc_df['asr'])
    weights = [asr/sum(acc_df['asr'].tolist()) for asr in acc_df['asr'].tolist()]
 

    for indx in prompt_name_score.index:
        # prompt_name_score_result['max50_score'][indx] = pd.to_numeric(prompt_name_score.iloc[indx][1:]).T.nlargest(50).T.mean()
        # prompt_name_score_result['max10_score'][indx] = pd.to_numeric(prompt_name_score.iloc[indx][1:]).T.nlargest(10).T.mean()
        # prompt_name_score_result['max5_score'][indx] = pd.to_numeric(prompt_name_score.iloc[indx][1:]).T.nlargest(5).T.mean()
        # prompt_name_score_result['max3_score'][indx] = pd.to_numeric(prompt_name_score.iloc[indx][1:]).T.nlargest(3).T.mean()
        # prompt_name_score_result['min5_score'][indx] = pd.to_numeric(prompt_name_score.iloc[indx][1:]).T.nsmallest(5).T.mean()
        # prompt_name_score_result['min3_score'][indx] = pd.to_numeric(prompt_name_score.iloc[indx][1:]).T.nsmallest(3).T.mean()
        weighted_scores = [score*weight for score, weight in zip(prompt_name_score.iloc[indx][1:].tolist(), weights) ]
        prompt_name_score_result['weighted_score'][indx] = sum(weighted_scores)/len(weighted_scores)


    
    prompt_name_score_result['is_in_train'] = [1 for i in range(num_pos_name)] + [0 for i in range(num_neg_name)]

    mv = []
    oracle = [] 
    for in_train in prompt_name_score.values.tolist()[:num_pos_name]:
        for not_in_train in prompt_name_score.values.tolist()[num_pos_name:]:
            # print(in_train)
            # print(len(not_in_train))
            cal = np.array(in_train) - np.array(not_in_train)
            wins = len([win for win in cal if win > 0])
            # print(wins)
            if wins > 200:
                mv.append(1)
            else:
                mv.append(0)
            
            if wins > 0:
                oracle.append(1)
            else:
                oracle.append(0)
    
    print('mv:', len(mv), round(sum(mv)/len(mv)*100, 2))
    print('oracle:', len(oracle), round(sum(oracle)/len(oracle)*100,2))

    # mv_by_thred = []
    # votes = []
    # for indx in prompt_name_rank.index:
    #     ranks =  prompt_name_rank.iloc[indx].values.tolist()[1:]
    #     vote = 0
    #     for r in ranks:
    #         if r< (num_name/2):
    #             vote +=1
    #     mv_by_thred.append(int(vote>200))
    #     votes.append(vote)
    # prompt_name_rank_result = pd.concat((prompt_name_rank_result, pd.DataFrame({'votes': votes})), axis=1)
    # prompt_name_rank_result = pd.concat((prompt_name_rank_result, pd.DataFrame({'mv': mv_by_thred})), axis=1)
    # prompt_name_rank_result['is_in_train'] = [1 for i in range(num_pos_name)] + [0 for i in range(num_neg_name)]
    

    
    # return prompt_name_rank, prompt_name_score, prompt_name_rank_result, prompt_name_score_result

    return prompt_name_rank, prompt_name_score, prompt_name_score_result

