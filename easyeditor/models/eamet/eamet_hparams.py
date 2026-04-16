from dataclasses import dataclass, field
from typing import List, Literal, Union

import yaml

from ...util.hparams import HyperParams


@dataclass
class EAMETHyperParams(HyperParams):
    alg_name: str
    model_name: str
    device: int
    stats_dir: str

            
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
                                             
    mom2_update_weight: Union[float, List[float]]

                      
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

                
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False

                    
    ks_bs: int = 64
    cs_factor: float = 2.0
    mse_factor: float = 8.0
    tau: float = 0.07
    norm_factor: float = 0.0
    clamp_ks_factor: float = 2.0
    delta_init: str = "target_init"
    weight_decay_method: str = "fix_ks"

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        assert config and config['alg_name'] == 'EAMET'
        return cls(**config)
