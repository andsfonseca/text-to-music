import torch

def create_device():
    """ 
    Selects and returns the appropriate device for PyTorch operations. Prioritizes the selection 
    of devices in the following order: CUDA > MPS > CPU.

    Returns:
        torch.device: The selected device for PyTorch operations.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        try:
            device = torch.device('mps')

            single_tensor_test = torch.tensor(0, device=device)
            assert(single_tensor_test.device == device)
            
            return device
        except Exception as e:
            print(f"MPS test failed with error: {e}. Using CPU instead")
    else:
        return torch.device('cpu')