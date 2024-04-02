import torch
import torchvision
from torchvision import datasets, models, transforms

def number_of_params(model):
    trainable_params = 0
    non_trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            param =  torch.flatten(param)
            trainable_params += param.shape[0]
        else:
            param =  torch.flatten(param)
            non_trainable_params += param.shape[0]

    print(f'Trainable paramters: {trainable_params:,d}')
    print(f'Non-trainable paramters: {non_trainable_params}')
