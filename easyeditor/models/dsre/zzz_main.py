from copy import deepcopy
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from .ZZZ import ZZZ
from .router import KnowRouter
from .utils import get_context_templates, tokenize
from .zzz_hparams import ZZZHyperParams


def apply_zzz_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: ZZZHyperParams,
        router: KnowRouter,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    if copy:
        model = deepcopy(model)
    device = f'cuda:{hparams.device}'

    context_templates = get_context_templates(model, tok, length_params=[[5, 5], [10, 5]], device=device)
    request = requests[0]

    ffn_id = router.route(request['prompt'])

    editor = ZZZ(
        model=model,
        config=hparams,
        device=device,
        router=router,
    )

    tokens, act_mask, deact_mask = tokenize(
        requests,
        tokenizer=tok,
        device=device,
        context_templates=context_templates,
        hparams=hparams
    )

    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask, prompt=request['prompt'])

    weights_copy = None
    return editor, weights_copy
