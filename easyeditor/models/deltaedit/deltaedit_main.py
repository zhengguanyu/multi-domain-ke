







from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

                                                                            
from ..memit.compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .deltaedit_hparams import DeltaEditHyperParams

                   
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

                                          
_weight_init = None
_U_trunks = {}
_cache_ks = {}
_case_ids = 0


def apply_deltaedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DeltaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_deltaedit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (upd_matrix,) in deltas.items():
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix.to(w.device)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    return model, weights_copy


def execute_deltaedit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DeltaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:

    global _weight_init, _U_trunks, _cache_ks, _case_ids

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

    if _weight_init is None:
        _weight_init = {k: v.detach().clone() for k, v in weights.items()}

    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

                               
    context_templates = get_context_templates(model, tok, hparams.device)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cur_z = compute_z(model, tok, request, hparams, z_layer,
                          context_templates, _case_ids)
        z_list.append(cur_z)

    zs = torch.stack(z_list, dim=1)

                                                                      
    for i, layer in enumerate(hparams.layers[:-1]):
        print(f"\n\nLAYER {layer}\n")

        layer_ks = compute_ks(model, tok, requests, hparams, layer,
                              context_templates).T

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
                      hparams.mom2_dtype, device=hparams.device,
                      stats_dir=hparams.stats_dir)

        layer_ks, targets = layer_ks.to(device), targets.to(device)

                                                         
        if _U_trunks.get(layer) is None:
            U, S, V = torch.linalg.svd(cov)
            small_indices = (S < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
            U_trunk = U[:, small_indices] @ U[:, small_indices].T
            _U_trunks[layer] = U_trunk
        else:
            U_trunk = _U_trunks[layer]

        if _cache_ks.get(layer) is None:
            _cache_ks[layer] = torch.zeros(
                (layer_ks.size(0), layer_ks.size(0)), device=device)

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        resid = targets / (len(hparams.layers) - i - 1)

        adj_k = torch.linalg.solve(
            U_trunk.to(device).T @ (layer_ks @ layer_ks.T + _cache_ks[layer]) +
            hparams.L2 * torch.eye(layer_ks.shape[0], device=device),
            U_trunk.to(device).T @ layer_ks
        )
        upd_matrix = resid @ adj_k.T

        _cache_ks[layer] += layer_ks @ layer_ks.T

                                                                  
        from .compute_z import cache_adj
        if layer == z_layer:
            upd_clone = upd_matrix.detach().clone()
            if cache_adj.get(layer) is None:
                cache_adj[layer] = upd_clone
            else:
                cache_adj[layer] += upd_clone

        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (upd_matrix.detach().cpu(),)

        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    _case_ids += 1

                                                               
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    return deltas


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
    else:
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
