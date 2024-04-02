import torch
import torchvision
from torchvision import datasets, models, transforms
from torchprofile import profile_macs



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


def FLOPs(model, inputs=None):
    #provide the input of the model    
    if inputs is None:
        inputs = torch.randn(1, 3, 224, 224)
    macs = profile_macs(model, inputs)

    print(f'FLOPs(G): {macs/1000000000:.3f}')



