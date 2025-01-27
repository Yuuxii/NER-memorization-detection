# Prompt_recomendation

The source code of the paper: Exploring Prompts to Elicit Memorization in Masked Language Model-based Named Entity Recognition.


## Data collection

We provide the code for data collection under  `data_collection/` directory. Note the provided code is for collecting organization and location eneties.

The collected entities and generated prompts are stored under `entities` directory.

## Memorization Detection Analysis

Source code is under `src/` directory.

Main function to detection memorization of NER: `exp_v1_group_prompts.py`.

Corelation and significant analysis: `data_analysis.ipynb`.

Prompt engeering and token-level analysis: `prompt_engeering_token_level_analysis.py`.

Prompt emsemble: `prompt_ensemble.py`.

Model attention heatmap analysis: `create_heatmaps.py`.
