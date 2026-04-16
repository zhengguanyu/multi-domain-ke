







from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

                                                                                     
                        
from ..memit.compute_ks import compute_ks
from ..alphaedit.compute_z import compute_z, get_module_input_output_at_words
from .NDEdit_hparams import NDEditHyperParams
from .rewrited_mlp import apply_rewrited_mlp, GatedKVMixin

                   
CONTEXT_TEMPLATES_CACHE = None


def apply_ndedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NDEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Any]:






    if copy:
        model = deepcopy(model)

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or\
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    context_templates = get_context_templates(model, tok, hparams.device)
    z_layer = hparams.layers[-1]
    device = f"cuda:{hparams.device}"

                                                       
    z_list = []
    for request in requests:
        cur_z = compute_z(model, tok, request, hparams, z_layer, context_templates)
        z_list.append(cur_z)

    zs = torch.stack(z_list, dim=1).detach().T                            

                                                                     
    for layer in hparams.layers:
        print(f"\n\nLAYER {layer}\n")

        layer_ks_list, cur_zs_list = [], []
        bs = hparams.ndedit_batch_size

        for j in range(0, len(requests), bs):
            batch_requests = requests[j:j + bs]
            batch_layer_ks = compute_ks(
                model, tok, batch_requests, hparams, layer, context_templates
            ).detach()
            batch_cur_zs = get_module_input_output_at_words(
                model, tok, z_layer,
                context_templates=[r["prompt"] for r in batch_requests],
                words=[r["subject"] for r in batch_requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].detach()
            layer_ks_list.append(batch_layer_ks)
            cur_zs_list.append(batch_cur_zs)

        layer_ks = torch.cat(layer_ks_list, dim=0)
        cur_zs = torch.cat(cur_zs_list, dim=0)
        target_v = zs - cur_zs

        print(f"Writing {layer_ks.size(0)} key/value pair(s) into layer {layer}")

                                     
        cur_mlp = nethook.get_module(model, hparams.mlp_module_tmp.format(layer))

        if isinstance(cur_mlp, GatedKVMixin):
                                                                      
            cur_mlp.update_KV(layer_ks, target_v)
        else:
                                                         
            try:
                new_mlp = apply_rewrited_mlp(cur_mlp, model.config)
            except Exception:
                new_mlp = apply_rewrited_mlp(cur_mlp, model.transformer.config)
            new_mlp.update_KV(layer_ks, target_v)
            cur_layer_module = nethook.get_module(
                model, hparams.layer_module_tmp.format(layer))
            setattr(cur_layer_module, 'mlp', new_mlp)

        del layer_ks, cur_zs, target_v
        torch.cuda.empty_cache()

                                                                        
                                                                           
    weights_copy = {}

    return model, weights_copy


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
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
