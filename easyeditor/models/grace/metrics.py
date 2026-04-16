import torch
import numpy as np
from .utils import *

def is_acc_error(model, tokens):
                                                                                
    labels = tokens["labels"]
    logits = model(**tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return labels != argmaxs

def Accuracy(model, tokens):
    labels = tokens["labels"]
    new_tokens = {f"{k}" : v for k, v in tokens.items() if k != "labels"}
    logits = model(**new_tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return (labels == argmaxs).float().mean()

def is_qa_error(model, tokens):
    preds = model.generate(tokens["input_ids"], max_length=20).squeeze()                                   
    labels = tokens["labels"]                           

    if (len(preds) != len(labels)) or ((preds == labels).sum() != len(preds)):
        return True
    else:
        return False

def PPL(model, batch):
    input_ids = batch["input_ids"][:, :1024]            
    if "labels" not in batch:
        target_ids = batch["input_ids"][:, :1024].clone()
    else:
        target_ids = batch["labels"][:, :1024].clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=target_ids)
        nll = outputs.loss

    ppl = torch.exp(nll)              
    return ppl

def F1(model, batch):
    try:
        preds = model.generate(batch["input_ids"], max_length=20).squeeze()
        if len(preds) > 1:
            preds = preds[preds != model.tokenizer.pad_token_id]
        gold_toks = batch["labels"][batch["labels"] != -100].cpu().squeeze()                         
        num_same = len(np.intersect1d(preds.cpu().squeeze(), gold_toks))
        if (num_same == 0) or (len(preds.squeeze()) == 0):
            return 0
        precision = num_same / len(preds.squeeze())
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    except:
                                                                      
        return 0
