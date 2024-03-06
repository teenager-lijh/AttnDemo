import torch


# def try_gpu():
#     if torch.backends.mps.is_available():
#         device = 'mps'
#     elif torch.cuda.is_available():
#         device = 'cuda:3'
#     else:
#         device = 'cpu'
#
#     return torch.device(device)


def try_gpu():
    if torch.cuda.is_available():
        device = 'cuda:3'
    else:
        device = 'cpu'

    return torch.device(device)
