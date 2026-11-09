"""
Logit lens experiment

NOTE: code in this file is designed for models from LLaMA-2 family. Its application to models with different architecture will require changing the code appropriately. 
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import random
import re
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import pickle

import random
import json
from itertools import permutations

random.seed(42)
np.random.seed(42)

device = "cuda:0"
option_ids = list('ABCDEF')
options_to_permute = list('ABCD')

datasets_names = ['mmlu_10k', 'hellaswag_10k', 'halu_dialogue_10k', 'cosmosqa_10k']

model_name = 'Llama-2-7b-hf'

if model_name in ['Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-hf']:
  model = LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}",
                                          output_hidden_states = True,
                                          torch_dtype=torch.float16
                                          )
  tokenizer = LlamaTokenizer.from_pretrained(f"meta-llama/{model_name}",
  )
else:
    raise ValueError("This script is designed for LLaMA2 models. Please provide appropriate changes to the code.")
    
model = model.to(device)

def create_user_prompt(example, options, option_ids = list('ABCDEF'), prompt = ''):
        user_prompt = f"Question: {example['question'].strip()}\nOptions:\n" + \
        "\n".join([f"{option_id}. {answer}".strip()
                    for option_id, answer in options.items()]) + \
        "\nAnswer:"
        if 'context' in example.keys():
                user_prompt = "Context: " + example['context'] + '\n' + user_prompt
        if prompt != '':
                user_prompt =  prompt + user_prompt
        return user_prompt

def prepare_prompts(example, options_ids, options_to_permute, k):
        options_dict = example['choices']
        answer = example['answer']
        if k == 1:
            return [options_dict], [answer]
        else:
            options_dict = example['choices']
            answer = example['answer']
            perms = [list(p) for p in random.sample(list(permutations(options_to_permute)), k = k-1)]
            perms = [p + [o for o in options_ids if o not in options_to_permute] for p in perms]
            perms_mappings = [{o: p for o, p in zip(options_ids, perm)} for perm in perms]
            permuted_answer = [perm[answer] for perm in perms_mappings]
            permuted_choices = [{perm[k]:options_dict[k] for k in options_ids} for perm in perms_mappings]
            permuted_choices = [dict(sorted(p.items())) for p in permuted_choices]
            return [options_dict] + permuted_choices, [answer] + permuted_answer




def logit_lens(hidden_states, model, n):
    if n < 32:
        return model.lm_head(model.model.norm(hidden_states[n])).squeeze().detach().cpu().numpy()[-1, :]
    else:
        return model.lm_head(hidden_states[n]).squeeze().detach().cpu().numpy()[-1, :]

def do_calc(data, model, tokenizer, prompt, option_ids, options_to_permute, k):
    option_indices = [tokenizer(f': {e}').input_ids[-1] for e in option_ids]
    labels_to_ids = {'A': 0, 'B': 1, 'C': 2, 'D':3, 'E': 4, 'F': 5}
    predictions = np.zeros((10000, k, 32))
    answers = np.zeros((10000, k))
    trunc_fraction = 0
    for EXMPL in tqdm(range(10000)):
        example = data[EXMPL]
        permuted_choices, permuted_answer = prepare_prompts(example, option_ids, options_to_permute, k)
        for i, (options, answer) in enumerate(zip(permuted_choices, permuted_answer)):
            input_text = create_user_prompt(example, options, prompt=prompt)
            #print(input_text)
            input_ids = tokenizer(input_text, truncation=False, return_tensors="pt").input_ids.to(device)
            if input_ids.shape[1] > 2300:
                input_ids = input_ids[:, -2300:]
                trunc_fraction += 1
            with torch.no_grad():
                hidden_states = model(input_ids = input_ids).hidden_states
            predictions[EXMPL, i] = [np.argmax(logit_lens(hidden_states, model, i)[option_indices]) for i in range(1, 33)]
            del hidden_states
            torch.cuda.empty_cache()
            answers[EXMPL, i] = labels_to_ids[answer]
    print(trunc_fraction / 10000)    
    return predictions, answers



for name in datasets_names:
  data_path = f'../data/{name}.json'

  with open(f'few_shot_prompts/{name.split("_10k_new")[0]}.json') as f:
    few_shot_prompts = json.load(f)

  for n_shot in [0,1,2,3,4,5]:

    prompt = few_shot_prompts[f'{n_shot}-shot']
    if len(option_ids) != 6:
      prompt = prompt.replace("\nE. I don't know .\nF. None of the above .", '')
    prompt = prompt.replace(' .', '.')

    with open(data_path) as json_data:
      data = json.load(json_data)

    predictions, answers = do_calc(data, model, tokenizer, prompt, option_ids, options_to_permute, k = 1)

    with open(f'results/{name}_{n_shot}-shot_logit_lens.npy', 'wb') as outfile:
        np.save(outfile, predictions)
    with open(f'results/{name}_{n_shot}-shot_perm_answers.npy', 'wb') as outfile:
        np.save(outfile, answers)