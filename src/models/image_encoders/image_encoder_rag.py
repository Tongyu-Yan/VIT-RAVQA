import pytorch_lightning as pl
import torch.nn as nn
from urllib.request import urlretrieve
import numpy as np
import os
import torch
from PIL import Image
import requests
import json
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionConfig

class image_encoder(pl.LightningModule):
    def __init__(self, config):
        super(image_encoder, self).__init__()
        self.config = config
        vit_config_ref = globals()[self.config.vit_model_config.VisionModelConfigClass]
        vit_config = vit_config_ref()
        vit_class = globals()[self.config.vit_model_config.VisionModelClass]
        #print(vit_class)
        self.vit = vit_class(vit_config)
        self.vit = self.vit.from_pretrained(self.config.vit_model_config.VisionModelVersion)
        for param in self.vit.parameters():
            param.requires_grad = False
        #self.map = nn.Linear(768, 1024)
        self.linear1 = nn.Linear(768, 1024 * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024 * 4, 1024 * 32)
        # self.q = nn.Linear(1024, 1024)
        self.k = nn.Linear(1024, 1024)
        self.v = nn.Linear(1024, 1024)

    def forward(self, x, text_embedding):
        x = self.vit(x)
        image_embedding = x.last_hidden_state[:,0]
        image_embedding = self.linear1(image_embedding)
        image_embedding = self.relu(image_embedding)
        image_embed = self.linear2(image_embedding)
        image_embed = image_embed.view(-1, 32, 1024)
        text_embed = text_embedding
        q = text_embed
        k = self.k(image_embed)
        v = self.v(image_embed)
        # Apply cross modal attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1))/32 #32 is sqrt(1024)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, v)
        return attended_features

    def save_pretrained(self, save_directory):
        """
        Save the image encoder model parameters to the specified directory.

        Args:
            save_directory (str): Directory to save the model parameters.
        """
        os.makedirs(save_directory, exist_ok=True)
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        # Save the configuration as well if it's necessary
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            # Assuming self.config is serializable to JSON
            json.dump(vars(self.config), f)

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

