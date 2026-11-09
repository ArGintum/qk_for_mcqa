# Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2410.02343-b31b1b.svg)](https://arxiv.org/abs/2410.02343)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Repository is under construction*

This is the official implementation of the NeurIPS 2026 paper:

> **Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models**
>
> Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Anastasia Voznyuk, Andrei Andriiainen, Irina Piontkovskaya, Evgeny Burnaev, Serguei Barannikov
>
> 

## Overview

LLMs often possess the internal knowledge to answer correctly to a multiple-choice question but fail to output it - tripped up by formatting quirks, biases toward certain option letters, or later layers overriding an earlier correct signal. We needed a way to read what the model internally knows rather than what it ultimately says. In this paper, we propose a method to open the black box and listen to model's attention heads individually. By measuring (at certain heads) the strength of alignment between Query and Key vectors - parts forming the attention mechanism, stripped-down from the positional information - we recover the model's latent choice in a single forward pass, no extra training and no output parsing. Across 24 models (1.5B–72B) from multiple families, this single-head readout beats the model's own zero-shot accuracy by up to 49.8 percentage points.

**Our contributions:**
- We show that **select-and-copy heads** are present in LLMs ranging from 1.5B to 70B parameters, performing the option selection operation for MCQA tasks, and the best of them are universal and consistent across datasets;
- We introduce the **Query-Key (QK) Score**, a novel option-scoring method based on key-query representations from these heads. Using it help to improve accuracy by 9-16\%, and also provides information about which model components contribute most to MCQA task solving; 
- We demonstrate that this method is more **stable** than baselines when handling option permutations or the addition of supplementary options (e.g., "I don't know");
- Our results support the hypothesis that **semantic representations are encoded in specific heads** in the query, key, and value vectors of a phrase's final tokens (namely, end-of-sentence or end-of-line tokens), as observed in previous studies;
- We analyse the attention patterns of *select-and-copy* heads and their behaviour under different conditions, **advancing our understanding of how attention mechanism** can work as ``select and copy'' operation, and how LLMs function in general.

## Abstract 

Large language models (LLMs) routinely fail to output the correct option in multiple-choice question answering (MCQA) while encoding the answer internally. We expose this latent knowledge via the Query--Key (QK) score, defined for an attention head as the inner product between the last-token query and the key at the end-of-line token following option $i$, evaluated before rotary positional embedding is applied. Its argmax identifies a universal class of select-and-copy heads in middle layers that perform option selection through semantic query--key alignment, mechanistically distinct from induction and copy-suppression heads (Olsson et al., 2022): they are invariant to label symbols, and solve a synthetic task with zero surface overlap---properties no positional-copy account explains and that critically require stripping RoPE.

Across 24 models from 1.5B to 72B parameters (LLaMA-2/3/3.1/3.3, Qwen-2.5, Gemma, Phi-3.5, DeepSeek-R1-Distill), a single head's QK-score exceeds the model's own zero-shot accuracy by up to $+27.4$ pp on HellaSwag and $+49.8$ pp on HaluDialogue; causal zero-ablation collapses MCQA accuracy to near-random. To remove any dependence on labeled validation data, we introduce an unsupervised HeadScore that ranks heads from unlabeled inputs and recovers the supervised top-$k$ heads on every tested model. Against four positional-debiasing baselines (e.g., PriDe, Wiegrefe, Wang), QK-score is complementary by construction: debiasing re-weights output logits, whereas QK-score reads the model's selection from a middle-layer head before decoding. Here we release a one-line drop-in HeadScore script and per-model head indices, making every result one-command reproducible across all 24 models and four benchmarks. 

## Project Structure

```
qk_for_mcqa/
├── data/                                          # Datasets storage
│   ├── synthetic_QA/															 # Synthetic dataset (Section 6.1)
│   │   ├── synthetic_QA_2_plus_additional_options_200_examples.json
│   │   ├── synthetic_QA_4_plus_additional_options_200_examples.json
│   │   ├── synthetic_QA_10_plus_additional_options_200_examples.json
│   │   ├── synthetic_QA_15_plus_additional_options_200_examples.json
│   │   ├── synthetic_QA_20_plus_additional_options_200_examples.json
│   │   └── synthetic_QA_24_plus_additional_options_200_examples.json
│   ├── cosmosqa_10k.json
│   ├── halu_dialogue_10k.json
│   ├── hellaswag_10k.json
│   └── mmlu_10k.json
├── scripts/                        # Scripts for baselines and auxiliary experiments
│   ├── att_stat_run.py             # Test different option-representative tokens for attention baseline
│   ├── get_pride_preditions.py     # Reproduce the results for PriDe baseline
│   ├── get_logit_lens.py           # Reproduce the Logit Lens experiment
│   ├── heads_ablation.py           # Reproduce the head ablation experiment (section 7.3) - QK-selected heads
│   ├── heads_ablation_random.py    # Reproduce the head ablation experiment (section 7.3) - random control
│   ├── prompt.py                   # Sets of in-context examples for experiments
│   ├── qk_stat_run.py              # Test different option-representative tokens for QK-score
│   ├── utils.py
│   └── utils_debias.py
├── Example_main_experiments.ipynb  # Notebook with code to reproduce our main experiments
├── utils_main.py 
├── prompt.py                       # Sets of in-context examples for the main experiments
├── qk_mcqa_scheme.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://https://github.com/ArGintum/qk_for_mcqa.git
```

### 2. Prepare the auxiliary folders

/////

```bash
cd qk_for_mcqa
mkdir cache
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```python

```

## Reproducing Results

### 1111

**Step 1:**


```bash 
```

**Step 2:**

```bash
```


## Citation

If you find this work useful, please cite our paper (this article was accepted for NeurIPS 2026; this is a preprint citation -- we will update the reference after the Conference):

```bibtex
@article{qk_mcqa2026tulchinskii,
  title={Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models},
  author={Tulchinskii, Eduard and Kuznetsov, Kristian and Kushanreva, Laida and Voznyuk, Anastasia and Andriiainen, Andrei and Piontkovskaya, Irina and Burnaev, Evgeny and Barannikov, Serguei},
  journal={arXiv preprint arXiv:2410.02343},
  year={2026}
}
``` 
    

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
