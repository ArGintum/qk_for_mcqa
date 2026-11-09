from collections import defaultdict
import numpy as np
import pandas as pd
import random
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pickle
import torch.nn.functional as F


import random
import json
from typing import List
from itertools import permutations

def create_user_prompt(question: str, options: List[str], option_ids: List[str], context = '', prompt = ''):
        user_prompt = f"Question: {question.strip()}\nOptions:\n" + \
        "\n".join([f"{option_id}. {answer}".strip()
                    for option_id, answer in zip(option_ids, options)]) + \
        "\nAnswer:"
        if context != '':
                user_prompt = "Context: " + context + '\n' + user_prompt
        if prompt != '':
                user_prompt =  prompt + user_prompt
        return user_prompt



def get_dev_set(data_path):
        df = pd.read_json(f'{data_path}')
        df = df[df.split == 'dev']
        df = pd.concat([df, df.choices.apply(pd.Series)], axis = 1)
        return df

def cycle_options(answers, answers_to_add):
    n = len(answers)
    for i in range(n):
        cycled_answers = answers[i:] + answers[:i] + answers_to_add
        yield cycled_answers


def full_permute_options(answers, answers_to_add, k):
    sampled_answers = [list(el) for el in random.sample(list(permutations(answers)), k = k)]
    for i in range(k):
        sampled_answers[i] += answers_to_add
    return sampled_answers

def factorial(num):
    if num == 0:
        return 1
    return num * factorial(num - 1)


def prepare_prompts(df, prompt, options_ids, options_to_permute):
        for i, row in df.iterrows():

                options = list(cycle_options([row[e] for e in options_to_permute],
                                            [row[e] for e in options_ids if e not in options_to_permute]))
                
                question = row['question']
                if 'context' in row.keys():
                        context = row['context']
                else:
                        context = ''
                
                user_prompts = []
                for o in options:
                        user_prompt = create_user_prompt(question, o, options_ids, context, prompt)
                        user_prompts.append(user_prompt)
                if i == 0:
                       print("Prompt example: \n", user_prompt)
                yield user_prompts

def get_observed(model, tokenizer, input_text, option_ids, option_indices, device):
        input_ids = tokenizer(input_text, truncation=False, return_tensors="pt").input_ids.to(device)
        #input_ids = input_ids[..., -1536:]
        with torch.no_grad():
                outputs = model(input_ids = input_ids)
        
        logits = outputs.logits.detach().cpu()
        logits = logits[:, -1, :]
        logits_full = logits.squeeze(0)
        logits_reduced = logits_full[option_indices].numpy()
        
        probs = softmax(logits_reduced)
        #probs = probs.reshape(input_ids.size(0), 2, len(option_ids)).sum(axis=1)
        return probs

def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x

def get_prior(df, prompt, model, tokenizer, option_ids, options_to_permute, device = 'cuda:1', return_observed = False):
        
        # option_indices = [tokenizer(f': {e}').input_ids[-1] for e in option_ids] + \
        #         [tokenizer(f':{e}').input_ids[-1] for e in option_ids]
        option_indices = [tokenizer(f': {e}').input_ids[-1] for e in option_ids]
        print(option_indices)
        prior = np.zeros((1, len(option_ids)))
        observed_total = []
        for prompts_permuted in tqdm(prepare_prompts(df, prompt, option_ids, options_to_permute), total = len(df)):
                prior_single = np.zeros((1, len(option_ids)))
                observed_temp = np.zeros((len(option_ids), len(option_ids)))
                for i, input_text in enumerate(prompts_permuted):
                        observed = get_observed(model, tokenizer, input_text, option_ids, option_indices, device)
                        observed_temp[i] = observed
                        prior_single += np.log(observed + 1e-10)
                prior_single = softmax(prior_single / len(prompts_permuted))

                prior += prior_single

                observed_total.append(observed_temp)

        prior =  prior[0] / len(df)
        observed_total = np.array(observed_total)
        if return_observed:
               return prior, observed_total
        else: 
               return prior