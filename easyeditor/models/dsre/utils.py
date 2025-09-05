import transformers
import torch
import os
import struct


CONTEXT_TEMPLATES_CACHE = None


def find_sublist_start_index(list1, list2):
    for i in range(len(list1) - len(list2)+1):
        if all(a == b for a, b in zip(list1[i:i+len(list2)], list2)):
            return i
    return None

class EarlyStopMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.pre = 0
        self.val = 1e9
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.pre = self.val
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def stop(self, ):
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02


class EditingMeanAct:

    def __init__(self, min_a=1e9):
        self.reset(min_a=min_a)

    def reset(self, min_a=1e9):
        self.avg = 0
        self.count = 0
        self.sum = 0
        self.min_a = min_a

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
        self.min_a = min(self.min_a, val)

    def mean_act(self):
        return self.avg
    def min_act(self):
        return self.min_a

def get_context_templates(model, tok, length_params, device):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = []
        prompt_tok = tok(
            ["I", "You", "Because", 'Yes', 'Q: '],
            padding=True,
            return_tensors="pt"
        ).to(device)
        for length, n_gen in length_params:

            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tok.eos_token_id,
            )
            CONTEXT_TEMPLATES_CACHE += tok.batch_decode(gen_token, skip_special_tokens=True)
        CONTEXT_TEMPLATES_CACHE = ['{}'] + [_ + ' {}' for _ in CONTEXT_TEMPLATES_CACHE]

    return CONTEXT_TEMPLATES_CACHE


def tokenize(batch, tokenizer, device, context_templates=None, hparams=None):
    len_temp = len(context_templates)
    prompts = [item['prompt'] for item in batch]
    labels = [item['target_new'] for item in batch]
    
    mask_token = -100
    if hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
        full_prompt = [tokenizer.apply_chat_template([{"role": "user", "content": templ.format(p)}],
                                                     add_generation_prompt=True,
                                                     tokenize=False) + ' ' + l
                       for templ in context_templates for p, l in zip(prompts, labels)]
        prompt_ids = tokenizer([tokenizer.apply_chat_template([{"role": "user", "content": templ.format(p)}],
                                                              add_generation_prompt=True,
                                                              tokenize=False) for templ in context_templates for p in prompts], return_tensors="pt", padding=True, truncation=True)["input_ids"]
    else:
        full_prompt = [f"{templ.format(p + ' ' + l)}" for templ in context_templates for p, l in zip(prompts, labels)]
        prompt_ids = tokenizer([f"{templ.format(p)}" for templ in context_templates for p in prompts], return_tensors="pt", padding=True, truncation=True)["input_ids"]
     
    num_prompt_toks = [len(i) for i in prompt_ids]
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token
    act_masks = []
    deact_masks = []

    act_masks = [mask.to(device) if mask is not None else None for mask in act_masks]
    deact_masks = [mask.to(device) if mask is not None else None for mask in deact_masks]

    tokens = {key: val.to(device) for key, val in tokens.items()}
    return tokens, act_masks, deact_masks
