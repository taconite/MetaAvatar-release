import torch

def get_params_by_key(model, key, exclude=False):
    if exclude:
        for name, param in model.named_parameters():
            if name != key:
                yield param
    else:
        for name, param in model.named_parameters():
            if name == key:
                yield param
