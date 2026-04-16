




from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook
from .eamet_hparams import EAMETHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: EAMETHyperParams,
    layer: int,
    context_templates: List[str],
    delta_list: List[torch.Tensor],
    combine_weights: torch.Tensor,
    request_id: int,
    layer_ks_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = f"cuda:{hparams.device}"

    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        if lm_b is None:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    target_new_str = request["target_new"]
    target_ids = tok.encode(target_new_str, return_tensors="pt",
                            add_special_tokens=False).to(device)[0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    opt_target_ids = target_ids

    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]

    all_prompts = rewriting_prompts + kl_prompts
    subjects = [request['subject']] * len(rewriting_prompts) + [request['subject']]
    all_filled_prompts = [p.format(s) for p, s in zip(all_prompts, subjects)]

    input_tok = tok(all_filled_prompts, return_tensors="pt", padding=True).to(device)

    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(opt_target_ids): ex_len] = opt_target_ids

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, subject, tok, hparams.fact_token, verbose=False
        )
        for prompt, subject in zip(all_prompts, subjects)
    ]

    loss_layer = max(hparams.v_loss_layer, layer)
    hidden_dim = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'n_embd')
    delta = torch.zeros((hidden_dim,), device=device, requires_grad=True)
    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, delta
        if cur_layer == hparams.layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                if hparams.delta_init == "target_init":
                    if torch.norm(target_init) < 100:
                        init_data = target_init.data.clone().detach()
                        init_data = init_data / torch.norm(init_data) * layer_ks_norm
                        delta.data.copy_(init_data)
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    scaled_max_norm = hparams.clamp_ks_factor * layer_ks_norm

    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[hparams.layer_module_tmp.format(layer),
                    hparams.layer_module_tmp.format(loss_layer)],
            retain_input=False, retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            kl_logits = torch.stack(
                [logits[-len(kl_prompts) + i, idx, :]
                 for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            :len(rewriting_prompts)]
        after_mult = ln_f(full_repr) @ lm_w + lm_b
        log_probs = torch.log_softmax(after_mult, dim=2)
        loss_val = torch.gather(
            log_probs, 2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

                                                   
        sample_cs = combine_weights[request_id]
        if len(delta_list) == 0:
            collision_loss, mse_loss = 0, 0
        else:
            avoid_z_ind = list(range(len(delta_list)))
            collision_loss, mse_loss = keep_cs_structure(
                delta_list, avoid_z_ind, sample_cs, delta, tau=hparams.tau)
            collision_loss = hparams.cs_factor * collision_loss
            mse_loss = hparams.mse_factor * mse_loss

        nll_loss_each = -(loss_val * mask).sum(1) / opt_target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean")

        if hparams.weight_decay_method == "norm":
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init))
        elif hparams.weight_decay_method == "fix_ks":
            weight_decay = hparams.norm_factor * (torch.norm(delta) - scaled_max_norm) ** 2
        else:
            raise ValueError(f"Unknown weight_decay_method: {hparams.weight_decay_method}")

        total_loss = nll_loss + kl_loss + weight_decay + collision_loss + mse_loss

        if total_loss < 1e-2 and it > 0:
            break
        if it == hparams.v_num_grad_steps - 1:
            break

        total_loss.backward()
        opt.step()

                                
        if hparams.weight_decay_method == "fix_ks":
            max_norm = scaled_max_norm
        else:
            max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    return target.detach().cpu(), delta.detach().cpu()


def compute_ks_collision_score(layer_ks: torch.Tensor) -> torch.Tensor:

    normalized_ks = layer_ks / torch.norm(layer_ks, dim=0, keepdim=True)
    similarities = torch.mm(normalized_ks.T, normalized_ks)
    similarities.fill_diagonal_(0)
    return similarities


def topk_cs(similarities, k):
    k = min(k, similarities.size(0))
    _, indices = torch.topk(similarities, k)
    return indices.tolist()


def keep_cs_structure(delta_list, avoid_z_ind, k_similarity, delta,
                      tau=0.07, eps=1e-8, islog=False):

    r_new = F.normalize(delta, dim=0)
    r_candidates = torch.stack([delta_list[i] for i in avoid_z_ind], dim=0)
    r_candidates = F.normalize(r_candidates, dim=1)

    sim_r = torch.matmul(r_candidates, r_new)
    q = F.softmax(sim_r / tau, dim=0)

    sim_k = k_similarity[avoid_z_ind]
    p = F.softmax(sim_k / tau, dim=0)

    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))

    topk_idx = topk_cs(sim_k, 50)
    sim_k_topk = sim_k[topk_idx].detach()
    sim_r_topk = sim_r[topk_idx]
    p_topk = F.softmax(sim_k_topk / tau, dim=0)
    mse = torch.sum(p_topk * (sim_k_topk - sim_r_topk) ** 2)

    return kl, mse


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
    if fact_token_strategy == "last":
        return -1
    elif "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        return repr_tools.get_words_idxs_in_templates(
            tok=tok, context_templates=[prompt], words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")
