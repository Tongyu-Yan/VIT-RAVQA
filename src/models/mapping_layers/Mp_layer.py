import pytorch_lightning as pl
import torch.nn as nn
from urllib.request import urlretrieve
import numpy as np
import os
import torch
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPVisionModel, CLIPVisionConfig


# Define the MapVIT class
class MapVIT(pl.LightningModule):
    def __init__(self, config):
        super(MapVIT, self).__init__()
        self.config = config
        vit_class = self.config.vit_model_config.VisionModelClass
        self.vit = vit_class.from_pretrained(self.config.vit_model_config.VisionModelVersion)
        self.vit.eval()
        
        
        # Load the file with the absolute path
        self.vit.load_from(np.load(checkpoint_file))
        self.vit.eval()
        self.map = nn.Linear(768, 768)

    def forward(self, x):
        x = self.vit(x)
        return self.map(x)

# Rest of your code...
