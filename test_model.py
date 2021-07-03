from CNN.model import build_backbone
import torch
inputs = torch.randn(1, 3, 500, 500)
config = {'cfg': 'efficientnet_b8', 'include_top': False, 'out_indices': [3, 11, 19, 30, 41, 56, 60, 'final_conv']}
# config = {'cfg': 'efficientnet_b7', 'include_top': False, 'out_indices': [3, 10, 17, 27, 37, 50, 54, 'final_conv']}
# config = {'cfg': 'efficientnet_b6', 'include_top': False, 'out_indices': [2, 8, 14, 22, 30, 41, 44, 'final_conv']}
effi = build_backbone(config)
out = effi(inputs)
print(effi.out_channel)
print([o.shape for o in out])

