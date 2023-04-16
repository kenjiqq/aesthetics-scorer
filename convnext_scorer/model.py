import torch
import torch.nn as nn
import timm
from safetensors import safe_open
from safetensors.torch import save_file
import collections
import os
import json

class ConvnextAestheticsScorer(nn.Module):

    def __init__(self, model=None, pretrained=False, config=None, head_layer_count=1):
        super().__init__()
        self.config = {
            "base_model": model,
            "head_layer_count": head_layer_count
        }
        if config != None:
            self.config.update(config)
        
        self.model = timm.create_model(self.config["base_model"], pretrained=pretrained, num_classes=1)
        in_dim = self.model.head[-1].in_features
        mlp_layers = [nn.Linear(in_dim, in_dim), nn.ReLU()] * (self.config["head_layer_count"] - 1) + [nn.Linear(in_dim, 1)]
        mlp = nn.Sequential(*mlp_layers)
        new_head = list(self.model.head.named_children())[:-1] + [('fc', mlp)]
        self.model.head = nn.Sequential(collections.OrderedDict(new_head))



    def forward(self, images):
        out = self.model(images)
        return out
    
    def save(self, save_name):
        split_name = os.path.splitext(save_name)
        with open(f"{split_name[0]}.config", "w") as outfile:
            outfile.write(json.dumps(self.config, indent=4))

        save_file(self.state_dict(), save_name)


def load_model(path, new_head=False, new_head_layer_count=1):
    split_path = os.path.splitext(path)
    with open(f"{split_path[0]}.config", "r") as config_file:
        config = json.load(config_file)

    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
           weights[key] = f.get_tensor(key)


    if new_head == True:
        config["head_layer_count"] = new_head_layer_count
    
    model =  ConvnextAestheticsScorer(config=config)
    
    if new_head:
        print("Initializing new head!")
        weights = {key: model.state_dict()[key] if key.startswith("model.head") else weights[key] for key in model.state_dict().keys()}

    model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    #model = ConvnextAestheticsScorer("convnext_large", head_layer_count=1)
    model = load_model("convnext_scorer/fake_detector_model.safetensors", new_head=True, new_head_layer_count=1)
    #model.save("test.pt")