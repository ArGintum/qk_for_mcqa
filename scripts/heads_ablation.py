"""
Head-ablation study

NOTE: version of code in this file is designed for models from LLaMA-2 family. Its application to models with different architecture will require appropriate changes. 
"""

import utils
import utils_debias
import os
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
import random
import json
import torch
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

import llama

device = "cuda"

ablate_type = 'label'
n_to_ablate = 10
option_ids = list('ABCDEF')

options_names = ''.join(option_ids)


datasets_names = ['mmlu_10k', 'hellaswag_10k', 'halu_dialogue_10k', 'cosmosqa_10k']

model_name = 'Llama-2-7b-hf'

if model_name in ['Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-hf']:
  model = llama.LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}",
                                          torch_dtype=torch.float16#,
                                          )
  tokenizer = LlamaTokenizer.from_pretrained(f"meta-llama/{model_name}",
  )


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

def encode(example, tokenizer, prompt = '', ablate_type = 'content_eol'):
      encodinds_context_q = []
      if 'context' in example.keys():    # Some quesions are given without context
          encodinds_context_q.append(tokenizer(prompt + "Context: " + example['context'] + "\nQuestion: ", return_tensors="pt"))
      else:
          encodinds_context_q.append(tokenizer(prompt + "Question: ", return_tensors="pt"))

      q_start = encodinds_context_q[-1]['input_ids'].shape[1] - 1
      encodinds_context_q.append(tokenizer(example['question'],  return_tensors="pt"))
      q_end = q_start + encodinds_context_q[-1]['input_ids'].shape[1] - 2
      encodinds_context_q.append(tokenizer("\nOptions:\n",  return_tensors="pt"))

      encodinds_context_q = {
                  "input_ids" : torch.cat([x["input_ids"][..., 1:] for x in encodinds_context_q], 1),
                  "attention_mask" : torch.cat([x["attention_mask"][..., 1:] for x in encodinds_context_q], 1)
              }
      
          
      num_q = encodinds_context_q["input_ids"].shape[-1] - 1

      encodings_answ, options_answ = [], []  
      option_label = []
      """ 
      For some experiments we need to permute answer options
      """
      options_raw, answer_raw = example['choices'], example['answer']

      for option in options_raw.keys():
          options_raw[option] = str(options_raw[option])            
          encodings_answ.append(tokenizer(option + ". " + options_raw[option] + "\n", return_tensors="pt"))
          if len(options_answ) == 0:
              options_answ.append(int(num_q + encodings_answ[-1]["input_ids"].shape[-1] - 1))
              option_label.append(int(num_q + 1))
          else:
              options_answ.append(int(options_answ[-1] + encodings_answ[-1]["input_ids"].shape[-1] - 1))
              option_label.append(int(options_answ[-2] + 1))

      encodings_answ.append(tokenizer("Answer:", return_tensors="pt"))
      inputs = {
          "input_ids" : torch.cat([encodinds_context_q["input_ids"]] + [x["input_ids"][..., 1:] for x in encodings_answ], 1).to(device),
      }
      if inputs['input_ids'].shape[1] > 2300:
          inputs['input_ids'] = inputs['input_ids'][:, -2300:]

      if ablate_type == 'content_eol':
          return inputs, answer_raw, options_answ
      elif ablate_type == 'label':
          return inputs, answer_raw, option_label

def get_best_heads(df, model, tokenizer, prompt, k, ablate_type):
    predictions = []
    true_answers = []
    for i, example in tqdm(df.iterrows(), total = len(df)):
        label_to_id = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}
        inputs, answer_raw, options_answ = encode(example, tokenizer, prompt, ablate_type)
        true_answers.append(label_to_id[answer_raw])
        with torch.no_grad():
            outputs = model(inputs)
        preds = []
        for j in range(32):
            preds.append(outputs['attention'+str(j)][0,:,options_answ])
        predictions.append(np.stack(preds, axis=0))
        del outputs
        torch.cuda.empty_cache()
    accs = (np.argmax(predictions, axis = -1) == np.array(true_answers)[:, None, None]).mean(axis = 0)
    ls,hs = np.unravel_index(accs.argsort(axis=None), shape=accs.shape)
    heads = {}
    for l, h in zip(ls[-k:], hs[-k:]):
        if l in heads.keys():
            heads[l].append(h)
        else:
            heads[l] = [h]
    return heads


def register_head_hooks(model, layer_heads, hook_fn, index):
    """
    Register |hook_fn| on |model| for |layer_heads|.

    Parameters
    ----------
    model : required, AutoModelForCausalLM
        Language model.
    layer_heads : required, dict
        Dictionary of layer/heads to ablate.
    hook_fn : required, Callable
        Ablation hook function.
    index : required, int
        Position of the last token.

    Returns
    ------
    hooks : list
        List of registered forward hooks.
    """
    hooks = []
    for l, h in layer_heads.items():
            hooks += [
                model.model.layers[l].self_attn.k_proj.register_forward_hook(
                    hook_fn(h, index)
                ),
                model.model.layers[l].self_attn.v_proj.register_forward_hook(
                    hook_fn(h, index)
                ),
                model.model.layers[l].self_attn.q_proj.register_forward_hook(
                    hook_fn(h, index)
                ),
            ]
    return hooks

def zero_ablate_heads(heads: list, index: int):
    """
    Returns abblation hook function which zero-ablates |heads| at the
    final token.

    Parameters
    ----------
    heads : required, list
        List of attention heads to ablate.
    index : required, int
        Position of the last token.

    Returns
    ------
    hook : Callable
        Ablation hook function.
    """

    def hook(model, input, output):
        for h in heads:
            output[:, index, h * (128) : (h + 1) * 128] = 0
        return output

    return hook





for name in datasets_names:
    data_path = f'data/{name}.json'
    df = utils_debias.get_dev_set(data_path)
    #df = df.iloc[:5]

    with open(f'few_shot_prompts/{name.split("_10k")[0]}.json') as f:
        few_shot_prompts = json.load(f)

    results = {"accuracy": [],
              "ourmetric": [],
              'head_ablated': []}

    for n_shot in [0,1,2,3,4,5]:

        prompt = few_shot_prompts[f'{n_shot}-shot']
        if len(option_ids) != 6:
            prompt = prompt.replace("\nE. I don't know .\nF. None of the above .", '')
        prompt = prompt.replace(' .', '.')

        atthooksmodel = utils.AttHooksModel(model)
        heads = get_best_heads(df, atthooksmodel, tokenizer, prompt, n_to_ablate, ablate_type)

        for h in atthooksmodel.fhooks:
            h.remove()
        del atthooksmodel


        with open(data_path) as json_data:
            data = json.load(json_data)

        hooks = register_head_hooks(model, heads, zero_ablate_heads, -1)
        true_labels, next_token_labels = utils.do_calc_eval(model, tokenizer, data, prompt=prompt, samples_range=range(500, 10000), permute=False, option_ids=option_ids, device = device)
        true_labels2, next_token_labels2 = utils.do_calc_eval(model, tokenizer, data, prompt=prompt, samples_range=range(500, 10000), permute=True, option_ids=option_ids, device = device)

        acc = np.mean(np.array(true_labels) == next_token_labels)
        ours = utils.compute_PA(true_labels, next_token_labels, true_labels2, next_token_labels2)

        results['accuracy'].append(acc)
        results['ourmetric'].append(ours)
        #results['head_ablated'].append(heads)

        for h in hooks:
            h.remove()

    with open(f'results/{model_name}_{name}_{ablate_type}_k={n_to_ablate}.json', 'w') as f:
      json.dump(results, f)