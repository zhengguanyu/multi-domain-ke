




from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook
from .deltaedit_hparams import DeltaEditHyperParams

                                          
cache_adj = {}
cache_n = {}


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: DeltaEditHyperParams,
    layer: int,
    context_templates: List[str],
    case_ids: int = 0,
) -> torch.Tensor:

    device = f"cuda:{hparams.device}"

    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    target_new_str = request["target_new"]
    target_ids = tok.encode(target_new_str, return_tensors="pt",
                            add_special_tokens=False).to(device)[0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt", padding=True,
    ).to(device)

    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids): ex_len] = target_ids

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)

    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=device)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    else:
        raise NotImplementedError

    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    layer_ks = None
    sim_ids = None

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
                hparams.rewrite_module_tmp.format(layer),
            ],
            retain_input=True, retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            kl_logits = torch.stack(
                [logits[i - len(kl_prompts), idx, :]
                 for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

                                      
        cos_loss = torch.tensor(0.0, device=device)
        if layer_ks is None:
            layer_ks = tr[hparams.rewrite_module_tmp.format(layer)
                          ].input[0, lookup_idxs[0]].detach().clone()

        if sim_ids is None and cache_adj.get(layer) is not None:
            noise = cache_adj[layer] @ layer_ks
            noise_norm = noise.norm()

            if cache_n.get(layer) is None:
                cache_n[layer] = {"mean": noise_norm, "std": torch.tensor(0.0, device=device)}

            mean = cache_n[layer]["mean"]
            std = cache_n[layer]["std"]

            noise_warmup = getattr(hparams, 'noise_warmup', 20)
            if (case_ids > noise_warmup and std != 0 and
                    (noise_norm - mean).abs() > hparams.var * std.sqrt()):
                sim_ids = True
                cache = torch.cat([noise.view(-1, 1), cache_adj[layer]], dim=1)
                s_size = cache.shape[0]
                U, S, _ = torch.svd_lowrank(cache @ cache.T,
                                            q=int(s_size * hparams.space))
                threshold = 0.1
                small_singular_indices = (S > threshold).nonzero(as_tuple=True)[0]
                space_cows = int(s_size * (1 - hparams.space))
                I = torch.eye(s_size, device=device)
                if len(small_singular_indices) < space_cows:
                    P = I - U[:, small_singular_indices] @ U[:, small_singular_indices].T
                else:
                    P = I - U[:, :space_cows] @ U[:, :space_cows].T
            else:
                alpha = 0.9
                cache_n[layer]["mean"] = alpha * mean + (1 - alpha) * noise_norm
                diff_squared = (noise_norm - cache_n[layer]["mean"]) ** 2
                cache_n[layer]["std"] = alpha * std + (1 - alpha) * diff_squared
                sim_ids = False

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]

        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
            dim=2)
        loss = torch.gather(
            log_probs, 2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0
                        ).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean")
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2)

        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device) + cos_loss

        if loss < 5e-2:
            break
        if torch.exp(-nll_loss_each).mean().item() > 0.98:
            break
        if it == hparams.v_num_grad_steps - 1:
            break

        loss.backward()
        opt.step()

                                
        if sim_ids:
            with torch.no_grad():
                delta[...] = delta @ P

                                
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    return target


def get_module_input_output_at_words(
    model, tok, layer, context_templates, words,
    module_template, fact_token_strategy,
):
    word_repr_args = dict(
        model=model, tok=tok, layer=layer, module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_"):]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken,
            context_templates=context_templates, words=words,
            **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(prompt, subject, tok, fact_token_strategy, verbose=True):
    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok, context_templates=[prompt], words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if verbose:
        sentence = prompt.format(subject)
        print(f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
              tok.decode(tok(sentence)["input_ids"][ret]))
    return ret
