import os
import re
import torch
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, "found")


def extract_attention(p_pos, model, tokenizer, input_text):
    """
    :param p_pos: List with the position of the tokens we want to get the attention for
    :param model: The model from which we want to extract the attention
    :param tokenizer: Tokenizer used to encode the data
    :param input_text: The input sentence
    :return:
    """
    # Forward pass
    # inputs = tokenizer(input_text, return_tensors="pt").to(device)
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(input_text)).unsqueeze(0).to(device)
    model = model.to(device)
    with torch.no_grad():
        # output = model(**inputs, output_attentions=True)
        output = model(ids, output_attentions=True)
    # print(output)
    # Get attention weights
    # Shape: num_layers, num_heads, sequence_length, sequence_length
    attentions = torch.cat(output.attentions).to("cpu")
    # print(attentions.shape)

    # Permute to sequence_length, num_heads, num_layers, sequence_length
    attentions = attentions.permute(2, 1, 0, 3)
    # print("After permutation:", attentions.shape)

    # heads = len(attentions[0])
    # Get the attention for the specific tokens
    if len(p_pos) == 1:
        attentions_pos = attentions[p_pos]
    else:
        pos_cat = []
        for p in p_pos:
            pos_cat.append(attentions[p])
        attentions_pos = torch.stack(pos_cat).mean(dim=0)

    # print(attentions_pos.shape)
    return attentions_pos


def visualize_attention(attentions_pos, tok, plot_name):
    # Average over the number of heads
    avg_per_layer_attention = attentions_pos.mean(dim=0)
    avg_attention = torch.round(avg_per_layer_attention.mean(dim=0).unsqueeze(0), decimals=2).to("cpu")
    # avg_attention = avg_per_layer_attention.mean(dim=0).unsqueeze(0)
    # print(avg_attention.shape)
    sns.set(font_scale=3)
    
    hmp = sns.heatmap(avg_attention, vmin=0, vmax=0.25, annot=True,
                      linewidths=0.2, # linecolor="white",
                      cbar=True, square=True, xticklabels=tok, yticklabels=False)
    # plt.gca().collections[0].set_clim(torch.min(avg_attention),torch.max(avg_attention))
    # plt.savefig(plot_name + ".png")
    # figure = hmp.get_figure()
    # figure.savefig(plot_name + ".png", bbox_inches="tight")
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xticks(rotation=45)
    # fig.savefig('test2png.png', dpi=100)
    fig.savefig(plot_name, bbox_inches="tight", dpi=100)
    plt.clf()


def get_avg_attention(names, prompt, model, tokenizer, name_start, model_name):
    names_attentions = []
    for name in tqdm(names):
        # Calculate the position of the name in sentence
        if model_name == "bert":
            name_length = len(re.findall(r"\w+|[^\w\s]", name, re.UNICODE))
        else:
            name_length = len(tokenizer.tokenize(name))
        name_indices = [n for n in range(name_start, name_start + name_length)]

        # Add the name to the sentence
        text = prompt.replace("MASK", name)
        tokenized_input = tokenizer.tokenize(text)

        word_ids = tokenizer(text).word_ids()[1:-1]

        if model_name == "roberta" or model_name == "albert":
            token_pos = [i for i in range(name_start, name_start + name_length)]
        else:
            token_pos = [i for i, e in enumerate(word_ids) if e in name_indices]

        attention_weights = extract_attention(token_pos, model, tokenizer, tokenized_input)

        if name_start != 0:
            # Get the weights for the tokens before the name
            att_before = attention_weights[:, :, :name_start]
        # Get the weights for the tokens after the name
        att_after = attention_weights[:, :, token_pos[-1] + 1:]
        # Get the average of the attention weights of the name tokens
        token_pos_att = attention_weights[:, :, token_pos]
        mean_token_pos_att = torch.mean(token_pos_att, dim=2).unsqueeze(2)

        # Concat the attention weights of the sequence
        if name_start != 0:
            final_att = torch.concat([att_before, mean_token_pos_att, att_after], 2)
        else:
            final_att = torch.concat([mean_token_pos_att, att_after], 2)

        names_attentions.append(final_att)

    # Get the average over the names
    all_names_att = torch.stack(names_attentions, 3)
    mean_all_names_att = torch.mean(all_names_att, dim=3)

    avg_per_layer_attention = mean_all_names_att.mean(dim=0)
    avg_attention = torch.round(avg_per_layer_attention.mean(dim=0).unsqueeze(0), decimals=2).to("cpu")

    return avg_attention, tokenizer.tokenize(prompt.replace("MASK", "name"))
    # visualize_attention(mean_all_names_att,
    #                     tokenizer.tokenize(prompt.replace("MASK", "name")),
    #                     plot_name)


# Read the names
name_pairs_df = pd.read_csv("./pairwise_dataset/dev_names.csv")
name_pairs_df.head()

# Get the positive names
pos_names = name_pairs_df.pos_name.to_list()
# Get the negative names
neg_names = name_pairs_df.neg_name.to_list()
neg_names.remove("Reinhold Voß")
neg_names.remove("Adolf Indrebø")
# Get the union
all_names = pos_names + neg_names

print("Found", len(all_names), "names in total, from which", len(pos_names), "are positive and", len(neg_names),
      "are negative.")

# Define models and their attributes
models = {
    "albert_b": {"best_prompt": "Bravo, MASK, what an impressive performance!",
                       "worst_prompt": "Are you going to MASK’s art gallery opening tonight?",
                       "model_type": "ArBert/albert-base-v2-finetuned-ner",
                       "best_name_position": 2,
                       "worst_name_position": 4,
                       "best_punctuation": 1,
                       "worst_punctuation": 2,
                       "model_name": "albert",
                        "v_max":0.25
                       },
          "albert_l": {"best_prompt": "Oh, MASK, you’re a true gem in our team.",
                       "worst_prompt": "MASK, practice forgiveness towards yourself and others.",
                       "model_type": "Gladiator/albert-large-v2_ner_conll2003",
                       "best_name_position": 2,
                       "worst_name_position": 0,
                       "best_punctuation": 1,
                       "worst_punctuation": 1,
                       "model_name": "albert",
                        "v_max":0.3
                       },
          "bert_b": {"best_prompt": "Are you going to MASK's art gallery opening tonight?",
                     "worst_prompt": "Did MASK give you any advice on starting something new?",
                     "model_type": "dslim/bert-base-NER",
                     "best_name_position": 4,
                     "worst_name_position": 1,
                     "best_punctuation": 0,
                     "worst_punctuation": 0,
                     "model_name": "bert",
                        "v_max":0.25
                     },
          "bert_l": {"best_prompt": "What project is MASK working on?",
                     "worst_prompt": "I had a chance to meet MASK’s family.",
                     "model_type": "dslim/bert-large-NER",
                     "best_name_position": 3,
                     "worst_name_position": 6,
                     "best_punctuation": 0,
                     "worst_punctuation": 0,
                     "model_name": "bert",
                        "v_max":0.75
                     },
          "roberta_b": {"best_prompt": "MASK, can you recommend a good restaurant in town?",
                        "worst_prompt": "I had a great conversation with MASK at the party.",
                        "model_type": "dominiqueblok/roberta-base-finetuned-ner",
                        "best_name_position": 0,
                        "worst_name_position": 6,
                        "best_punctuation": 0,
                        "worst_punctuation": 0,
                        "model_name": "roberta",
                        "v_max":0.4
                        },
          "roberta_l": {"best_prompt": "MASK, invest in meaningful relationships.",
                        "worst_prompt": "MASK, practice playing the guitar.",
                        "model_type": "Gladiator/roberta-large_ner_conll2003",
                        "best_name_position": 0,
                        "worst_name_position": 0,
                        "best_punctuation": 0,
                        "worst_punctuation": 0,
                        "model_name": "roberta",
                        "v_max":0.5
                        }
          }

for m in models:
    m_info = models[m]
    print("Creating heatmaps for", m, "...")
    hmp_folder = m
    # if not os.path.exists(hmp_folder):
    #     os.makedirs(hmp_folder)

    tokenizer = AutoTokenizer.from_pretrained(m_info["model_type"])
    model = AutoModelForTokenClassification.from_pretrained(m_info["model_type"])
    
    # Best prompt with positive names
    best_pos_att, best_pos_tokens = get_avg_attention(pos_names, m_info["best_prompt"], model, tokenizer, m_info["best_name_position"],
                      m_info["model_name"])
    # Best prompt with negative names
    best_neg_att, best_neg_tokens = get_avg_attention(neg_names, m_info["best_prompt"], model, tokenizer, m_info["best_name_position"],
                      m_info["model_name"])
    # Best prompt with all names
    # get_avg_attention(all_names, m_info["best_prompt"], model, tokenizer, m_info["best_name_position"],
    #                   os.path.join(hmp_folder, "best_prompt_all_names.png"), m_info["model_name"])

    # Worst prompt with positive names
    worst_pos_att, worst_pos_tokens = get_avg_attention(pos_names, m_info["worst_prompt"], model, tokenizer, m_info["worst_name_position"],
                      m_info["model_name"])
    # Worst prompt with negative names
    worst_neg_att, worst_neg_tokens = get_avg_attention(neg_names, m_info["worst_prompt"], model, tokenizer, m_info["worst_name_position"],
                      m_info["model_name"])
    # Worst prompt with all names
    # get_avg_attention(all_names, m_info["worst_prompt"], model, tokenizer, m_info["worst_name_position"],
    #                   os.path.join(hmp_folder, "worst_prompt_all_names.png"), m_info["model_name"])
    
    
    sns.set(font_scale=1.8)
    
    fig, axn = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(20, 5))
    cbar_ax = fig.add_axes([.91, .3, .03, .5])

    attentions = [best_pos_att, worst_pos_att, best_neg_att, worst_neg_att]
    tokens = []
    for token in [best_pos_tokens, worst_pos_tokens, best_neg_tokens,  worst_neg_tokens]:
        modified_token = []
        if m in ['albert_b', 'albert_l', 'roberta_b', 'roberta_l']:
            for t in token:
                if len(t)>1:
                    if t[1:] == 'name':
                        modified_token.append('MASK')
                    else:
                        modified_token.append(t[1:])
                else:
                    modified_token.append(t)
        else:
            for t in token:
                if t == 'name':
                    modified_token.append('MASK')
                else:
                    modified_token.append(t)
        tokens.append(modified_token)
    # tokens = [best_pos_tokens, worst_pos_tokens, best_neg_tokens,  worst_neg_tokens]
    y_labels = [["In-train"], ["In-train"], ["Out-train"], ["Out-train"]]


    for idx, ax in enumerate(axn.flat):
        # print(idx)
        if idx == 0:
            cbar = True
        else:
            cbar = False
        g = sns.heatmap(attentions[idx], ax=ax, vmin=0, vmax=m_info["v_max"], annot=True,
                      linewidths=0.2, # linecolor="white",
                      cbar=cbar, square=True, xticklabels=False if idx in [0,1] else tokens[idx], yticklabels=False if idx in [1,3] else y_labels[idx],
                      cbar_ax=None if not cbar else cbar_ax, fmt='.2f', annot_kws={"fontsize":18})
        # ax.xaxis.tick_top()  # Put x axis on top
        # ax.set_aspect(aspect=1)
        if idx == 0:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel('Best prompt', fontsize = 25)
        if idx == 1:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel('Worst prompt', fontsize = 25)
        if idx not in [0,1]: 
            tl = ax.get_xticklabels()
            ax.set_xticklabels(tl, rotation=45)
            
        if idx not in [1,3]:
            tly = ax.get_yticklabels()
            ax.set_yticklabels(tly, rotation=0)
        # ax.tick_params(rotation=45)  # Do not rotate y tick labels
        
        # if ax != axes[0]:
        #     ax.set_yticks([])

    # plt.gca().collections[0].set_clim(torch.min(avg_attention),torch.max(avg_attention))
    # plt.savefig(plot_name + ".png")
    # figure = hmp.get_figure()
    # figure.savefig(plot_name + ".png", bbox_inches="tight")
    # axn[0][0].get_shared_y_axes().join(axn[0][1])
    # axn[1][0].get_shared_y_axes().join(axn[1][1])
    # axn[0][0].get_shared_x_axes().join(axn[1][0])
    # axn[0][1].get_shared_x_axes().join(axn[1][1])
    fig = plt.gcf()
    # fig.set_size_inches(28, 10.5)
    fig.tight_layout(rect=[0, 0, .9, 1])
  

    fig.savefig(os.path.join('heatmaps/', m+"_att_heatmap.png"), bbox_inches="tight", dpi=100)
    plt.clf()


    
    print("Done!")

