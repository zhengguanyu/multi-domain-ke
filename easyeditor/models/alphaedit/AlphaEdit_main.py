import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

                   
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
                                                  







    global P, P_loaded, cache_c, cache_c_new

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
                                                  
                                                                                                                                             
    if not os.path.exists(hparams.P_loc):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "llama" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower() or "qwen" in hparams.model_name.lower():
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model, tok, layer, hparams)
        torch.save(P, hparams.P_loc)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(hparams.P_loc)
        P_loaded = True

                                                                           
                                                                                                  
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "llama" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower() or "qwen" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        cache_c_new = True
    
    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(f"cuda:{hparams.device}")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:





    deltas = {}

                                  
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
                                                     
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or\
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

                                                  
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

                                             
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

                                                           
    context_templates = get_context_templates(model, tok)

    if getattr(hparams, 'blue', False):
                                                                  
        for i, layer in enumerate(hparams.layers):
            print(f"\n\nLAYER {layer}\n")
            z_layer = layer                                 

            z_list = []
            for request in requests:
                cache_fname = (
                    Path(
                        str(cache_template).format(
                            z_layer, hparams.clamp_norm_factor, request["case_id"]
                        )
                    )
                    if cache_template is not None
                    else None
                )
                data_loaded = False
                if cache_fname is not None and cache_fname.exists():
                    try:
                        data = np.load(cache_fname)
                        z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                        data_loaded = True
                    except Exception as e:
                        print(f"Error reading cache file due to {e}. Recomputing...")

                if not data_loaded:
                    cur_z = compute_z(
                        model, tok, request, hparams, z_layer, context_templates,
                    )
                    z_list.append(cur_z)
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(cache_fname, v_star=cur_z.detach().cpu().numpy())

            zs = torch.stack(z_list, dim=1)

            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            cur_zs = get_module_input_output_at_words(
                model, tok, z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs - cur_zs
            print("z error", torch.linalg.norm(targets, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            resid = targets                              

            dev = f"cuda:{hparams.device}"
            upd_matrix = torch.linalg.solve(
                P[i,:,:].to(dev) @ (layer_ks.to(dev) @ layer_ks.T.to(dev) + cache_c[i,:,:].to(dev)) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float, device=dev),
                P[i,:,:].to(dev) @ layer_ks.to(dev) @ resid.T.to(dev)
            )

            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            print("orig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
                deltas[weight_name] = upd_matrix.detach().cpu()

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()
    else:
                                                                          
        z_layer = hparams.layers[-1]
        z_list = []

        for request in requests:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        z_layer, hparams.clamp_norm_factor, request["case_id"]
                    )
                )
                if cache_template is not None
                else None
            )
            data_loaded = False
            if (
                cache_fname is not None
                and cache_fname.exists()
            ):
                try:
                    data = np.load(cache_fname)
                    z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            if not data_loaded:
                cur_z = compute_z(
                    model, tok, request, hparams, z_layer, context_templates,
                )
                z_list.append(cur_z)
                if cache_fname is not None:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(cache_fname, v_star=cur_z.detach().cpu().numpy())
                    print(f"Cached k/v pair at {cache_fname}")
        zs = torch.stack(z_list, dim=1)

        for i, layer in enumerate(hparams.layers):
            print(f"\n\nLAYER {layer}\n")

            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            cur_zs = get_module_input_output_at_words(
                model, tok, z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T
            targets = zs - cur_zs
            print("z error", torch.linalg.norm(targets, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            resid = targets / (len(hparams.layers) - i)                                     

            dev = f"cuda:{hparams.device}"
            upd_matrix = torch.linalg.solve(
                P[i,:,:].to(dev) @ (layer_ks.to(dev) @ layer_ks.T.to(dev) + cache_c[i,:,:].to(dev)) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float, device=dev),
                P[i,:,:].to(dev) @ layer_ks.to(dev) @ resid.T.to(dev)
            )

            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            print("orig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
                deltas[weight_name] = upd_matrix.detach().cpu()

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

                                     
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:





    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:





    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]                                   
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T