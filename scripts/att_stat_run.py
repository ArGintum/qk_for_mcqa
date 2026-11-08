from collections import defaultdict
import numpy as np
import pandas as pd
import random
import re
import argparse
from tqdm import tqdm
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import pickle
from prompt import *

import random
import json

random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='QK experiment setup')

parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Model to run experiments on, e.g. meta-llama/Llama-2-7b-hf")
parser.add_argument("--dataset", type=str, help="cosmosqa, halu, hellaswag or mmlu")
parser.add_argument("--n_shots", type=int, help="n-shots for promtp, [0..5]")
parser.add_argument("--n_test", type=int, default=10000, help="number of test examples")

args = parser.parse_args()

BASE_DIR = '../'
STORAGE_DIR = BASE_DIR + "cache/"
MODEL_NAME = parser.model
FILE_SAVE_NAME = MODEL_NAME.split("/")[-1]

model = LlamaForCausalLM.from_pretrained(MODEL_NAME, #"meta-llama/Llama-2-7b-hf",
                                         torch_dtype=torch.float16,
                                         output_attentions=True
                                        )
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME,#"meta-llama/Llama-2-7b-hf",
                                        )

LAYER_NUM = model.config.num_hidden_layers
HEAD_NUM = model.config.num_attention_heads

# Some architectures add fictitional first token, some doesn't;
# This constant tracks this fact.
FIRST_TOKEN_CORRECTION = len(tokenizer("A")['input_ids']) - 1


device = "cuda:0"
model = model.to(device)


def do_calc_attn(data, prompt='', samples_range=range(10000), permute=False):
    """
    Parameters:
        data      ------- self-explanatory
        prompt      ----- Here go examples in case of the Few-Shot prompting. For Zero-shot leave it empty.
        samples_range --- Container with numbers of samples to be considered 
        permute     ----- Specifies if a permutation of answer options is required   
    """
    true_labels = []
    predicted_labels = []

    n = len(list(samples_range))

    scores_dict = {"last": np.zeros((n, LAYER_NUM, HEAD_NUM)),
                   "first": np.zeros((n, LAYER_NUM, HEAD_NUM)),
                   "question_content_avg":  np.zeros((n, LAYER_NUM, HEAD_NUM)),
                   "question_qmark": np.zeros((n, LAYER_NUM, HEAD_NUM)),
                   "question_eol": np.zeros((n, LAYER_NUM, HEAD_NUM)),
                   "option_label": np.zeros((n, LAYER_NUM, HEAD_NUM, 6)),
                   "option_label_period": np.zeros((n, LAYER_NUM, HEAD_NUM, 6)),
                   "option_content_avg": np.zeros((n, LAYER_NUM, HEAD_NUM, 6)),
                   "option_content_period": np.zeros((n, LAYER_NUM, HEAD_NUM, 6)),
                   "option_content_eol": np.zeros((n, LAYER_NUM, HEAD_NUM, 6))}

    for LAYER in range(LAYER_NUM):
        predicted_labels.append([])

        for HEAD in range(HEAD_NUM):
            predicted_labels[LAYER].append([])

    for EXMPL in tqdm(samples_range):
        """
        Assembling the prompt from different parts: Examples (if any) + Context + Question + Options + Finisher
        """    
        encodinds_context_q = []
        if 'context' in data[EXMPL].keys():    # Some quesions are given without context
            encodinds_context_q.append(tokenizer(prompt + "Context: " + data[EXMPL]['context'] + "\nQuestion: ", return_tensors="pt"))
        else:
            encodinds_context_q.append(tokenizer(prompt + "Question: ", return_tensors="pt"))

        q_start = encodinds_context_q[-1]['input_ids'].shape[1] - 1
        encodinds_context_q.append(tokenizer(data[EXMPL]['question'],  return_tensors="pt"))
        q_end = q_start + encodinds_context_q[-1]['input_ids'].shape[1] - 1 - FIRST_TOKEN_CORRECTION
        encodinds_context_q.append(tokenizer("\nOptions:\n",  return_tensors="pt"))

        encodinds_context_q = {
                    "input_ids" : torch.cat([x["input_ids"][..., FIRST_TOKEN_CORRECTION:] for x in encodinds_context_q], 1),
                    "attention_mask" : torch.cat([x["attention_mask"][..., FIRST_TOKEN_CORRECTION:] for x in encodinds_context_q], 1)
                }
        
            
        num_q = encodinds_context_q["input_ids"].shape[-1] - 1

        encodings_answ, options_answ = [], []  
        option_label = []
        """ 
        For some experiments we need to permute answer options
        """
        options_raw, answer_raw = data[EXMPL]['choices'], data[EXMPL]['answer']

        for option in options_raw.keys():
            options_raw[option] = str(options_raw[option])            
            encodings_answ.append(tokenizer(option + ". " + options_raw[option] + "\n", return_tensors="pt"))
            if len(options_answ) == 0:
                options_answ.append(int(num_q + encodings_answ[-1]["input_ids"].shape[-1] - FIRST_TOKEN_CORRECTION))
                option_label.append(int(num_q + 1))
            else:
                options_answ.append(int(options_answ[-1] + encodings_answ[-1]["input_ids"].shape[-1] - FIRST_TOKEN_CORRECTION))
                option_label.append(int(options_answ[-2] + 1))

        encodings_answ.append(tokenizer("Answer:", return_tensors="pt"))
        inputs = {
            "input_ids" : torch.cat([encodinds_context_q["input_ids"]] + [x["input_ids"][..., FIRST_TOKEN_CORRECTION:] for x in encodings_answ], 1).to(device),
            "attention_mask" : torch.cat([encodinds_context_q["attention_mask"]] + [x["attention_mask"][..., FIRST_TOKEN_CORRECTION:] for x in encodings_answ], 1).to(device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs).attentions

        true_labels.append(answer_raw)

        for LAYER in range(LAYER_NUM):
            for HEAD in range(HEAD_NUM):
                predicts = np.zeros(len(options_answ))
                attn_last_row = outputs[LAYER][0][HEAD][-1]
                for i in range(len(options_answ)):
                    predicts[i] = attn_last_row[options_answ[i]].cpu()
                    scores_dict['option_content_avg'][EXMPL, LAYER, HEAD, i] = attn_last_row[option_label[i]+2:options_answ[i]-1].cpu().mean()

                predicted_labels[LAYER][HEAD].append(chr(np.argmax(predicts) + ord('A')))

                scores_dict['option_content_eol'][EXMPL, LAYER, HEAD, :] = predicts
                scores_dict['last'][EXMPL, LAYER, HEAD] = attn_last_row[-1].cpu()
                scores_dict['first'][EXMPL, LAYER, HEAD] = attn_last_row[0].cpu()
                scores_dict['question_content_avg'][EXMPL, LAYER, HEAD] = attn_last_row[q_start:q_end].cpu().mean()
                scores_dict['question_qmark'][EXMPL, LAYER, HEAD] = attn_last_row[q_end].cpu()
                scores_dict['question_eol'][EXMPL, LAYER, HEAD] = attn_last_row[q_end+2].cpu()

                scores_dict['option_label'][EXMPL, LAYER, HEAD, :] = attn_last_row[option_label].cpu()
                scores_dict['option_label_period'][EXMPL, LAYER, HEAD, :] = attn_last_row[[i+1 for i in option_label]].cpu()
                scores_dict['option_content_period'][EXMPL, LAYER, HEAD, :] = attn_last_row[[i-1 for i in options_answ]].cpu()


    for LAYER in range(LAYER_NUM):
        predicted_labels[LAYER] = np.array(predicted_labels[LAYER])

    predicted_labels = np.stack(predicted_labels)
    return true_labels, predicted_labels, scores_dict


def save_results_test(dataset, shots, qk_label, true_label):
    dict1 = {
        'true_labels' : true_label,
        'attn_predictions' : qk_label,
    }
    with open(STORAGE_DIR + '{}_10k_{}_{}-shot_predictions.pckl'.format(dataset, FILE_SAVE_NAME, shots), 'wb') as handle:
        pickle.dump(dict1, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_scores(dataset, shots, scores_dict):
    with open(STORAGE_DIR + '{}_10k_{}_{}-shot_scores.pckl'.format(dataset, FILE_SAVE_NAME, shots), 'wb') as handle:
        pickle.dump(scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#------------------------------------------------------------------------------


dataset_path = {
    'cosmosqa': 'cosmosqa_10k.json', 
    'halu': 'halu_dialogue_10k.json', 
    'hellaswag': 'hellaswag_10k.json',
    'mmlu': 'mmlu_10k.json'
}
DATASET_DIR = 'data/'

dataset_prompts = {
    'cosmosqa': cosmos_prompt, 
    'halu': haludialogue_prompt, 
    'hellaswag': hellaswag_prompt,
    'mmlu': mmlu_prompt
}
prompt = dataset_prompts[args.dataset][str(args.n_shots) + '-shot']

print(args)
print(prompt)

json_file = BASE_DIR + DATASET_DIR + dataset_path[args.dataset] 

with open(json_file) as json_data:
    data = json.load(json_data)

print(len(data))

true_labels, predicted_labels, scores_dict = do_calc_attn(data, prompt, range(args.n_test))

save_results_test(args.dataset, args.n_shots, predicted_labels, true_labels)

save_scores(args.dataset, args.n_shots, scores_dict)
