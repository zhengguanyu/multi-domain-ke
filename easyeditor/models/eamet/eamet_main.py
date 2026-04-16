







from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import (compute_z, compute_ks_collision_score,
                        get_module_input_output_at_words)
from .eamet_hparams import EAMETHyperParams

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_eamet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EAMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_eamet(model, tok, requests, hparams,
                           cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            device = f"cuda:{hparams.device}"
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


def execute_eamet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EAMETHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:

    device = f"cuda:{hparams.device}"
    deltas = {}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or\
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
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

                                                            
    z_layer_ks = compute_ks(model, tok, requests, hparams, z_layer,
                            context_templates).T
    layer_ks_norm = torch.norm(z_layer_ks, dim=0)
    combine_weights = compute_ks_collision_score(z_layer_ks)
    z_layer_ks.cpu()
    del z_layer_ks
    torch.cuda.empty_cache()

                                                          
    z_list, delta_list = [], []
    for re_id, request in enumerate(requests):
        print(f"Optimizing z for request {re_id}/{len(requests)}")
        opt_zs, delta = compute_z(
            model, tok, request, hparams, z_layer, context_templates,
            delta_list, combine_weights, request_id=re_id,
            layer_ks_norm=layer_ks_norm[re_id],
        )
        z_list.append(opt_zs.to(device))
        delta_list.append(delta.to(device))

    zs = torch.stack(z_list, dim=1)
    del delta_list
    torch.cuda.empty_cache()

                                                    
    muw = hparams.mom2_update_weight
    if isinstance(muw, (int, float)):
        muw = [muw] * len(hparams.layers)

                                   
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        layer_ks = compute_ks(model, tok, requests, hparams, layer,
                              context_templates).T

                                                  
        temps = [r['prompt'] for r in requests]
        temp_words = [r["subject"] for r in requests]
        cur_zs = get_module_input_output_at_words(
            model, tok, z_layer,
            context_templates=temps, words=temp_words,
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

        cov_mat = muw[i] * cov.double() + (layer_ks @ layer_ks.T)
        adj_k = torch.inverse(cov_mat.to("cpu")).to(device) @ layer_ks

        resid = targets / (len(hparams.layers) - i)
        upd_matrix = resid @ adj_k.T

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())

        cov.cpu()
        for x in [layer_ks, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

                              
    with torch.no_grad():
        for k in weights:
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    return deltas


def get_cov(model, tok, layer_name, mom2_dataset, mom2_n_samples,
            mom2_dtype, device=0, stats_dir="./data/stats", force_recompute=False):
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model, tok, layer_name, stats_dir, mom2_dataset,
            to_collect=["mom2"], sample_size=mom2_n_samples,
            precision=mom2_dtype, force_recompute=force_recompute,
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
