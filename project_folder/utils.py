import torch
import random

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_out_channels(in_channels: int, param_val) -> int:
    if isinstance(param_val, int):
        return param_val
    
    if isinstance(param_val, str):
        if param_val == 'same':
            return in_channels
        
        if 'sameX' in param_val:
            try:
                factor = int(param_val.split('X')[1])
                return in_channels * factor
            except (IndexError, ValueError):
                pass
        
        if 'same/' in param_val:
            try:
                divisor = int(param_val.split('/')[1])
                return max(1, in_channels // divisor)
            except (IndexError, ValueError):
                pass

    return in_channels