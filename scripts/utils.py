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
from collections import OrderedDict 


import random
import json

import torch.nn.functional as F


def permute_options(options, label, seed):
    """
    A pseudo-random shuffle of the answer <<options>> that also returns new correct answer
        parameters:
                options --- list of options
                label   --- correct answer
                seed    --- random seed
                
    Options 'E' and 'F' are special and must remain at their places
    """
    aaa = np.arange(len(options) - 2)
    np.random.seed(seed)
    np.random.shuffle(aaa)
    
    new_label = chr(ord('A') + aaa[ord(label) - ord('A')])
    new_options = {}
    for option in ['A', 'B', 'C', 'D']: 
        new_options[chr(aaa[ord(option) - ord('A')] + ord('A'))] = options[option]
   
    new_options['E'] = options['E']  
    new_options['F'] = options['F']

    return dict(sorted(new_options.items())), new_label

def qk_score(vec_a, vec_b):
    return torch.sum(vec_a * vec_b) 

def qk_score_matrix(mat_a, mat_b):
    return (mat_a @ mat_b.T) / (mat_a.norm(p=2, dim=-1).reshape((mat_a.shape[0], 1)) @ mat_b.norm(p=2, dim=-1).reshape((1, mat_b.shape[0])))



class NewModel(torch.nn.Module):
    def __init__(self, model, *args):
        super().__init__(*args)
        self.selected_out = OrderedDict()

        self.pretrained = model
        self.fhooks = []

        for i in range(model.config.num_hidden_layers):
            self.fhooks.append(self.pretrained.model.layers[i].self_attn.q_proj
                .register_forward_hook(self.forward_hook("query_vec_" + str(i))))
            self.fhooks.append(self.pretrained.model.layers[i].self_attn.k_proj
                .register_forward_hook(self.forward_hook("key_vec_" + str(i))))
        
        #    Value vectors were not used in the main experiments, but they can be considered too
        #    self.fhooks.append(self.pretrained.model.layers[i].self_attn.v_proj
        #        .register_forward_hook(self.forward_hook("value_vec_" + str(i))))
    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output.cpu()
        return hook

    def forward(self, x):        
        out = self.pretrained(**x)
        return out, self.selected_out
    

class AttHooksModel(torch.nn.Module):
    def __init__(self, model, *args):
        super().__init__(*args)
        self.selected_out = OrderedDict()

        self.pretrained = model
        self.fhooks = []

        for i in range(model.config.num_hidden_layers):
            self.fhooks.append(self.pretrained.model.layers[i].self_attn.attention
                .register_forward_hook(self.forward_hook("attention" + str(i))))
    
    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output.cpu()
        return hook

    def forward(self, x):        
        out = self.pretrained(**x)
        return self.selected_out

def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x


def do_calc_eval(model, tokenizer, data, option_ids=list('ABCDEF'), prior = None, prompt='', samples_range=range(10000), permute=False, device = 'cuda:1'):
    """
    Parameters:
        data      ------- self-explanatory
        head          --- Pair (#LAYER, #HEAD),
        prompt      ----- Here go examples in case of the Few-Shot prompting. For Zero-shot leave it empty.
        samples_range --- Container with numbers of samples to be considered 
        permute     ----- Specifies if a permutation of answer options is required    
    """
    FIRST_TOKEN_CORRECTION = len(tokenizer("A")['input_ids']) - 1

    true_labels = []
    next_token_labels = []
    debiased_next_token_labels = []

    for EXMPL in tqdm(samples_range):
        """
        Assembling the prompt from different parts: Examples (if any) + Context + Question + Options + Finisher
        """    
        if 'context' in data[EXMPL].keys():    # Some quesions are given without context
            encodinds_context_q = tokenizer(prompt + "Context: " + data[EXMPL]['context'] + "\nQuestion: " + \
                                            data[EXMPL]['question'] + "\nOptions:\n", return_tensors="pt")
        else:
            encodinds_context_q = tokenizer(prompt + "Question: " + data[EXMPL]['question'] + "\nOptions:\n", 
                                            return_tensors="pt")
            
        num_q = encodinds_context_q["input_ids"].shape[-1] - 1
        encodings_answ, options_answ = [], []
        
        """ 
        For some experiments we need to permute answer options
        """
        options_raw, answer_raw = data[EXMPL]['choices'], data[EXMPL]['answer']
        if permute:
            options_raw, answer_raw = permute_options(options_raw, answer_raw, EXMPL)
        
        for option in option_ids:
            options_raw[option] = str(options_raw[option])            
            encodings_answ.append(tokenizer(option + ". " + options_raw[option] + "\n", return_tensors="pt"))
            if len(options_answ) == 0:
                options_answ.append(int(num_q + encodings_answ[-1]["input_ids"].shape[-1] - FIRST_TOKEN_CORRECTION))
            else:
                options_answ.append(int(options_answ[-1] + encodings_answ[-1]["input_ids"].shape[-1] - FIRST_TOKEN_CORRECTION))

        encodings_answ.append(tokenizer("Answer:", return_tensors="pt"))
        inputs = {
            "input_ids" : torch.cat([encodinds_context_q["input_ids"]] + [x["input_ids"][..., FIRST_TOKEN_CORRECTION:] for x in encodings_answ], 1).to(device)
        }
        true_labels.append(answer_raw)
        
        """
        If promt format was changed use this to debug:
        
        print(inputs)
        for i in range(len(options_answ)):
            print(inputs["input_ids"][..., options_answ[i]]) # <<<<<< This must be aligned
        print("\n\n", inputs["input_ids"][..., -1])
        """
        with torch.no_grad():
            outputs = model(**inputs)
            # option_indices = [tokenizer(f': {e}').input_ids[-1] for e in option_ids] + \
            #     [tokenizer(f':{e}').input_ids[-1] for e in option_ids]
        option_indices = [tokenizer(f': {e}').input_ids[-1] for e in option_ids]
        logits = outputs.logits.detach().cpu()
        logits = logits[:, -1, :]
        logits_full = logits.squeeze(0)
        logits_reduced = logits_full[option_indices].numpy()

        del outputs
        torch.cuda.empty_cache()

        probs = softmax(logits_reduced)
        #probs = probs.reshape(input_ids.size(0), 2, len(option_ids)).sum(axis=1).squeeze()
        
        next_token_labels.append(chr(ord('A') + np.argmax(logits_reduced)))

        if prior is not None:
            assert prior.shape == probs.shape
            debiased_probs = np.log(probs + 1e-10) - np.log(prior + 1e-10)
            debiased_next_token_labels.append(chr(ord('A') + np.argmax(debiased_probs)))
            return np.array(true_labels), np.array(next_token_labels), np.array(debiased_next_token_labels)
    if prior is not None:
        return np.array(true_labels), np.array(next_token_labels), np.array(debiased_next_token_labels)
    else:
        return np.array(true_labels), np.array(next_token_labels)


def compute_PA(target, predictions, target_permuted, predictions_permuted):
    our_metric_value = np.mean((target == predictions) * (target_permuted == predictions_permuted))
    return our_metric_value