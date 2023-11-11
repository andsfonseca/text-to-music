import torch

def create_device():
    """
        Creates a torch device obeying the 
        following priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    # mps is breaking during training    
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        return torch.device('cpu')