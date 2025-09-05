from dataclasses import dataclass
from typing import List, Optional, Dict

import yaml
from omegaconf import DictConfig

from ...util.hparams import HyperParams


@dataclass
class ZZZHyperParams(HyperParams):
    alg_name: str
    model_name: str
    device: int
    inner_params: List[str]
    seed: int
    representation_layer_index: int
    embedding: Dict
    clustering: Dict
    mask_ratio: float
    edit_lr: float
    n_iter: int
    norm_constraint: float
    objective_optimization: str
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    two_stages: bool = True
    boundary_model_name: str = "None"
    boundary_threshold: float = 0.5
    use_clustering: bool = True
    use_multi_ffn: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
    
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'DSRE'), \
            f'ZZZHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'

        from omegaconf import OmegaConf
        config['embedding'] = OmegaConf.create(config['embedding'])
        config['clustering'] = OmegaConf.create(config['clustering'])

        return cls(**config)        
    