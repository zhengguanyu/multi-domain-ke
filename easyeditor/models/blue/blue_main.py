










from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

                                        
from ..memit.compute_z import (compute_z, get_module_input_output_at_words,
                                find_fact_lookup_idx)
from ..memit.compute_ks import compute_ks
from ..memit.memit_hparams import MEMITHyperParams
from .blue_hparams import BLUEHyperParams

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

                                      
_blue_cache_c = None

                           
_rect_cache_c = None


def apply_blue_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: BLUEHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_blue(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            device = f"cuda:{hparams.device}"
            upd_matrix = key_mat.to(device) @ val_mat.to(device).T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    return model, weights_copy


def apply_prune_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: BLUEHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

                                          
    deltas = execute_memit_standard(model, tok, requests, hparams)

    k_percent = hparams.k_percent
    epsilon = 1e-8

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            device = f"cuda:{hparams.device}"
            upd_matrix = key_mat.to(device) @ val_mat.to(device).T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

                                                                           
            delta_rel = torch.abs(upd_matrix / (w + epsilon))
            threshold = torch.kthvalue(
                delta_rel.reshape(-1),
                int(delta_rel.numel() * (100 - k_percent) / 100)
            ).values
            mask = delta_rel >= threshold
            w[mask] += upd_matrix[mask].float()

    return model, weights_copy


def apply_rect_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: BLUEHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas, updated_cache_c = execute_rect(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            device = f"cuda:{hparams.device}"
            upd_matrix = key_mat.to(device) @ val_mat.to(device).T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    return model, weights_copy


                                                                              
                          
                                                                              

def execute_blue(model, tok, requests, hparams):

    device = f"cuda:{hparams.device}"
    deltas = {}

    requests = deepcopy(requests)
    for i, r in enumerate(requests):
        if r["target_new"][0] != " ":
            requests[i]["target_new"] = " " + r["target_new"]
        if '{}' not in r['prompt']:
            assert r['subject'] in r['prompt'] or\
                   print(f"Subject:{r['subject']} do not exist in prompt: {r['prompt']}")
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok, hparams.device)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nBLUE LAYER {layer}\n")

                                                        
        z_layer = layer
        z_list = []
        for request in requests:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
        zs = torch.stack(z_list, dim=1)

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        targets = zs - cur_zs
        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        cov = get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                      hparams.mom2_dataset, hparams.mom2_n_samples,
                      hparams.mom2_dtype, hparams.device, hparams.stats_dir)

        layer_ks, targets = layer_ks.double().to(device), targets.double().to(device)

                                                   
        if torch.isnan(layer_ks).any() or torch.isinf(layer_ks).any() or\
           torch.isnan(targets).any() or torch.isinf(targets).any():
            print(f"WARNING: NaN/inf in layer_ks or targets at layer {layer}, skipping update")
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            cov.cpu()
            torch.cuda.empty_cache()
            continue

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T, layer_ks)

                                             
        resid = targets
        upd_matrix = resid @ adj_k.T

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

        cov.cpu()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for k in weights:
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    return deltas


def execute_memit_standard(model, tok, requests, hparams):

    device = f"cuda:{hparams.device}"
    deltas = {}

    requests = deepcopy(requests)
    for i, r in enumerate(requests):
        if r["target_new"][0] != " ":
            requests[i]["target_new"] = " " + r["target_new"]
        if '{}' not in r['prompt']:
            assert r['subject'] in r['prompt'] or\
                   print(f"Subject:{r['subject']} do not exist in prompt: {r['prompt']}")
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok, hparams.device)
    z_layer = hparams.layers[-1]

    z_list = []
    for request in requests:
        cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        targets = zs - cur_zs
        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        cov = get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                      hparams.mom2_dataset, hparams.mom2_n_samples,
                      hparams.mom2_dtype, hparams.device, hparams.stats_dir)

        layer_ks, targets = layer_ks.double().to(device), targets.double().to(device)
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T, layer_ks)
        resid = targets / (len(hparams.layers) - i)
        upd_matrix = resid @ adj_k.T

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

        cov.cpu()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for k in weights:
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    return deltas


def execute_rect(model, tok, requests, hparams):



    global _rect_cache_c
    device = f"cuda:{hparams.device}"
    deltas = {}
    use_blue = getattr(hparams, 'blue', False)

    requests = deepcopy(requests)
    for i, r in enumerate(requests):
        if r["target_new"][0] != " ":
            requests[i]["target_new"] = " " + r["target_new"]
        if '{}' not in r['prompt']:
            assert r['subject'] in r['prompt'] or\
                   print(f"Subject:{r['subject']} do not exist in prompt: {r['prompt']}")
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = get_context_templates(model, tok, hparams.device)

                                      
    if _rect_cache_c is None:
        W_out = nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        dim = W_out.shape[1] if "llama" in hparams.model_name.lower() or "qwen" in hparams.model_name.lower() else W_out.shape[0]
        _rect_cache_c = torch.zeros((len(hparams.layers), dim, dim))

    if not use_blue:
                                                
        z_layer = hparams.layers[-1]
        z_list = []
        for request in requests:
            cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
            z_list.append(cur_z)
        zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        if use_blue:
                                            
            z_layer = layer
            z_list = []
            for request in requests:
                cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
                z_list.append(cur_z)
            zs = torch.stack(z_list, dim=1)

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        targets = zs - cur_zs
        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        cov = get_cov(model, tok, hparams.rewrite_module_tmp.format(layer),
                      hparams.mom2_dataset, hparams.mom2_n_samples,
                      hparams.mom2_dtype, hparams.device, hparams.stats_dir)

        layer_ks, targets = layer_ks.double().to(device), targets.double().to(device)

                                                
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() +
            _rect_cache_c[i].to(device).double() +
            layer_ks @ layer_ks.T,
            layer_ks
        )

        if use_blue:
            resid = targets                              
        else:
            resid = targets / (len(hparams.layers) - i)

        upd_matrix = resid @ adj_k.T

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

                                  
        _rect_cache_c[i] += (layer_ks.cpu() @ layer_ks.cpu().T).float()

        cov.cpu()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for k in weights:
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    return deltas, _rect_cache_c


                                                                              
                  
                                                                              

def get_cov(model, tok, layer_name, mom2_dataset, mom2_n_samples,
            mom2_dtype, device=0, stats_dir="./data/stats", force_recompute=False):
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if key not in COV_CACHE or force_recompute:
                                                            
        class _Hparams:
            pass
        _hp = _Hparams()
        _hp.device = device
        stat = layer_stats(
            model, tok, layer_name, stats_dir, mom2_dataset,
            to_collect=["mom2"], sample_size=mom2_n_samples,
            precision=mom2_dtype, force_recompute=force_recompute,
            hparams=_hp,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return COV_CACHE[key].to(f"cuda:{device}")


def upd_matrix_match_shape(matrix, shape):
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    raise ValueError("Update matrix shape mismatch")


def get_context_templates(model, tok, device=0):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model, tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
    return CONTEXT_TEMPLATES_CACHE
