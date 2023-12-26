import torch

def save_checkpoint(model,epoch,optimizer):
    if hasattr(model,"module"):
        model = model.module
    checkpoint = {
        'net': weight_to_cpu(model.state_dict()),
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, "checkpoint.pth")

def weight_to_cpu(state_dict):
    cpu_state_dict = {}
    for key, val in state_dict.items():
        cpu_state_dict[key] = val.cpu()
    return cpu_state_dict