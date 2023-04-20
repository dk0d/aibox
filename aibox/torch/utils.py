try:
    import torch
    from torchvision import transforms
except ImportError:
    print("pytorch required for these utilities")
    exit(1)



def get_device(): 
    if torch.has_cuda:
        return 'cuda'
    if torch.has_mps: 
        return torch.device('mps')
        
    return torch.device('cpu')
