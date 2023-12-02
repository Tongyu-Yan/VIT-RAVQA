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
        vit_config_ref = globals()[self.config.vit_model_config.VisionModelConfigClass]
        vit_config = vit_config_ref()
        vit_class = globals()[self.config.vit_model_config.VisionModelClass]
        #print(vit_class)
        self.vit = vit_class(vit_config)
        self.vit = self.vit.from_pretrained(self.config.vit_model_config.VisionModelVersion)
        #self.vit = vit_class.from_pretrained(self.config.vit_model_config.VisionModelVersion)
        self.vit.eval()
        self.map = nn.Linear(768, 768)

    def forward(self, x):
        x = self.vit(x)
        return self.map(x)

# Rest of your code...
