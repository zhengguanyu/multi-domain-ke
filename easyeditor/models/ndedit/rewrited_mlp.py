





import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from transformers.models.llama.modeling_llama import LlamaMLP

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
except ImportError:
    GPT2MLP = None

try:
    from transformers.models.gptj.modeling_gptj import GPTJMLP
except ImportError:
    GPTJMLP = None

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
except ImportError:
    Qwen2MLP = None


def cos_sim(x, y):
    dot_product = x @ y.T
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    return dot_product / (x_norm * y_norm.T + 1e-8)


class GatedKVMixin:


    def init_kv(self):
        self.edited = True
        self.K = None
        self.V = None

    def update_KV(self, K, V):
        if self.K is None and self.V is None:
            self.K, self.V = K, V
        elif self.K is not None and self.V is not None:
            self.K = torch.cat([self.K, K], dim=0)
            self.V = torch.cat([self.V, V], dim=0)

    def compute_delta(self, h):
        assert self.K is not None and self.V is not None
        sim = cos_sim(h, self.K)
        indices = torch.zeros_like(sim, device=sim.device)
        max_indices = torch.max(sim, dim=-1)[1]
        indices.scatter_(-1, max_indices.unsqueeze(-1), 1.0)
        gate_in = (sim.max(-1)[0] > 0.65)
        gate = gate_in.unsqueeze(-1) * indices
        delta = torch.matmul(gate, self.V)
        return delta


class CustomLlamaMLP(LlamaMLP, GatedKVMixin):
    def __init__(self, original_mlp, config):
        super().__init__(config)
        self.load_state_dict(original_mlp.state_dict())
        self.to(original_mlp.down_proj.weight.device)
        self.init_kv()

    def forward(self, x):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(h) + self.compute_delta(h)
        return down_proj


                                            
class CustomQwen2MLP(GatedKVMixin, nn.Module):
    def __init__(self, original_mlp, config):
        super().__init__()
                                           
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        self.init_kv()

    def forward(self, x):
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(h) + self.compute_delta(h)
        return down_proj


if GPT2MLP is not None:
    class CustomGPT2MLP(GPT2MLP, GatedKVMixin):
        def __init__(self, original_mlp, config):
            inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
            super().__init__(intermediate_size=inner_dim, config=config)
            self.load_state_dict(original_mlp.state_dict())
            self.to(original_mlp.c_fc.weight.device)
            self.init_kv()

        def forward(self, hidden_states):
            hidden_states = self.c_fc(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(hidden_states) + self.compute_delta(hidden_states)
            hidden_states = self.dropout(hidden_states)
            return hidden_states


if GPTJMLP is not None:
    class CustomGPTJMLP(GPTJMLP, GatedKVMixin):
        def __init__(self, original_mlp, config):
            inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
            super().__init__(intermediate_size=inner_dim, config=config)
            self.load_state_dict(original_mlp.state_dict())
            self.to(original_mlp.fc_out.weight.device)
            self.init_kv()

        def forward(self, hidden_states):
            hidden_states = self.fc_in(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.fc_out(hidden_states) + self.compute_delta(hidden_states)
            hidden_states = self.dropout(hidden_states)
            return hidden_states


def apply_rewrited_mlp(old_mlp, config):

    if isinstance(old_mlp, LlamaMLP):
        return CustomLlamaMLP(old_mlp, config)
    if Qwen2MLP is not None and isinstance(old_mlp, Qwen2MLP):
        return CustomQwen2MLP(old_mlp, config)
    if GPT2MLP is not None and isinstance(old_mlp, GPT2MLP):
        return CustomGPT2MLP(old_mlp, config)
    if GPTJMLP is not None and isinstance(old_mlp, GPTJMLP):
        return CustomGPTJMLP(old_mlp, config)
                                                                 
    if hasattr(old_mlp, 'gate_proj') and hasattr(old_mlp, 'down_proj'):
        return CustomQwen2MLP(old_mlp, config)
    raise ValueError(f"Unsupported MLP type: {type(old_mlp)}")
