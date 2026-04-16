import torch


def dict_to(d, device):
    """Recursively move all torch.Tensor values in a dict to `device`.

    Extracted from the original easyeditor.trainer.utils so that dataset
    modules can use it without the deleted trainer package.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v
    return new_dict
