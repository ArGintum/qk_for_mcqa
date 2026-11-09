"""
Code for computing PRiDE baseline.
"""

import utils
import utils_debias
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import json
import torch
from prompt import * 

random.seed(42)
np.random.seed(42)

device = "cuda"
#device = "cpu"

option_ids = list('ABCDEF')
options_to_permute = list('ABCDEF')

options_names = ''.join(option_ids)

datasets_names = ['mmlu_10k', 'hellaswag_10k', 'halu_dialogue_10k', 'cosmosqa_10k']


model_name = 'Llama-2-7b-hf'
#model_name = 'Llama-2-7b-chat-hf'
#model_name = 'Llama-2-13b-hf'
model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}",
                                          torch_dtype=torch.float16#,
                                          )
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}",
                                         )


model = model.to(device)
model = utils.NewModel(model)

dataset_prompts = {
    'cosmosqa': cosmos_prompt, 
    'halu': haludialogue_prompt, 
    'hellaswag': hellaswag_prompt,
    'mmlu': mmlu_prompt
}


results = {}

for name in datasets_names:
    data_path = f'../data/{name}.json'
    df = utils_debias.get_dev_set(data_path)

    few_shot_prompts = dataset_prompts[name.split("_10k")[0]]
    for n_shot in [0,1,2,3,4,5]:

        prompt = few_shot_prompts[f'{n_shot}-shot']
        if len(option_ids) != 6:
            prompt = prompt.replace("\nE. I don't know .\nF. None of the above .", '')
        prompt = prompt.replace(' .', '.')

        prior, observed_total = utils_debias.get_prior(df, prompt, model, tokenizer, option_ids, options_to_permute, device, return_observed = True)
        print("Resulting prior", prior)

        with open(data_path) as json_data:
            data = json.load(json_data)

        true_labels, next_token_labels, debiased_next_token_labels = utils.do_calc_eval(model, tokenizer, data, prior = prior, prompt=prompt, samples_range=range(10000), permute=False, option_ids=option_ids, device = device)

        print(f'Dataset {name}, {n_shot}-shot')
        print(f"Baseline Acc: {(true_labels[500:] == next_token_labels[500:]).mean().round(3)}")
        print(f"Pride Acc: {(true_labels[500:] == debiased_next_token_labels[500:]).mean().round(3)}")

        with open(f'results/{name}_{options_names}_{n_shot}-shot_pride.npy', 'wb') as outfile:
            np.save(outfile, np.array(debiased_next_token_labels))

        with open(f'results/{name}_{options_names}_{n_shot}-shot_observed_for_pride.npy', 'wb') as outfile:
            np.save(outfile, np.array(observed_total))