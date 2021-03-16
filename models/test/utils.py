import os
import torch

def freeze(model,n_layers_to_freeze=10):
    modules = [model.embeddings, *model.encoder.layer[:n_layers_to_freeze]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

def save_model(model_to_save,save_directory,num_gpus=0):
    'Metode til at gemme en pytorch model, der tager højde for om modelen trænes i parallel eller ej'
    WEIGHTS_NAME = "pytorch_model.bin" # this comes from transformers.file_utils
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    if num_gpus > 1:
        model_to_save = model_to_save.module

    state_dict = model_to_save.state_dict()
    torch.save(state_dict, output_model_file)