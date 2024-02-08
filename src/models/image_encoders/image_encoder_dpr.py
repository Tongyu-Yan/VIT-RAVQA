import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from urllib.request import urlretrieve
import numpy as np
import os
import torch
from PIL import Image
import requests
import json
from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionConfig


# Define the MapVIT class
class Vis_encoder(pl.LightningModule):
    def __init__(self, config):
        super(Vis_encoder, self).__init__()
        self.config = config
        vit_config_ref = globals()[self.config.vit_model_config.VisionModelConfigClass]
        vit_config = vit_config_ref()
        vit_class = globals()[self.config.vit_model_config.VisionModelClass]
        #print(vit_class)
        self.vit = vit_class(vit_config)
        self.vit = self.vit.from_pretrained(self.config.vit_model_config.VisionModelVersion)
        #self.vit = vit_class.from_pretrained(self.config.vit_model_config.VisionModelVersion)
        #self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(768, 768 * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(768 * 4, 768)
        self.q = nn.Linear(768, 768)
        self.k = nn.Linear(768, 768)
        self.v = nn.Linear(768, 768)


    def forward(self, x, text_embedding):
        x = self.vit(x)
        image_embedding = x.last_hidden_state[:,0]
        image_embedding = self.linear1(image_embedding)
        image_embedding = self.relu(image_embedding)
        image_embed = self.linear2(image_embedding)
        text_embed = text_embedding
        q = self.q(image_embed)
        k = self.k(text_embed)
        v = self.v(image_embed)
        score = torch.einsum('ij,ij->ij', q, k) / np.sqrt(768)  # Element-wise multiplication and scaling
        attention_weights = F.sigmoid(score)  # Compute the attention weights
        weighted_v = attention_weights * v  # Apply the attention weights to the value
        return weighted_v

    def save_pretrained(self, save_directory):
        """
        Save MapVIT model parameters to the specified directory.

        Args:
        save_directory (str): Directory to save the model parameters.
        """
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Optionally, save the configuration if needed
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
    def from_pretrained(cls, save_directory):
        """
        Load a pretrained MapVIT model from the specified directory.

        Args:
        save_directory (str): Directory where the model parameters are saved.
        """
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = cls(config)
        
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        return model
"""
To save the model:
model = MapVIT(config)
# ... training and other operations ...

save_directory = '/path/to/save/directory'
model.save_pretrained(save_directory)

To load the model:
save_directory = '/path/to/saved/model'
model = MapVIT.from_pretrained(save_directory)


"""