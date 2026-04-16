









import contextlib
import copy
import inspect
from collections import OrderedDict

import torch


class Trace(contextlib.AbstractContextManager):






























    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):




        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )                                       
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                                                                
                                                                  
                                          
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):

















    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):












    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):





    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
                                                                   
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def subsequence(
    sequential,
    first_layer=None,
    last_layer=None,
    after_layer=None,
    upto_layer=None,
    single_layer=None,
    share_weights=False,
):












    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):






    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
                                        
                                                            
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:                              
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
                                                                   
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:                              
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
                                                            
                                      
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad, *models):




    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model, name):



    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model, name):



    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):



    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
                                                 
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):


















    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
                                                                   
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
                                                                          
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
                                                     
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
                                                             
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)
