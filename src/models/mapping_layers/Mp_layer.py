import pytorch_lightning as pl
import torch.nn as nn
from urllib.request import urlretrieve
import sys
sys.path.append('/home/ty308/rds/hpc-work/myvqa/Tony-VQA/src/models/VIT')

from modeling import VisionTransformer, CONFIGS
import numpy as np

import os

# Absolute paths should start with a slash if they are intended to be absolute
checkpoint_dir = '/home/ty308/rds/hpc_work/myvqa'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_file = os.path.join(checkpoint_dir, "ViT-B_16-224.npz")
if not os.path.isfile(checkpoint_file):
    urlretrieve(
        "https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
        checkpoint_file
    )
class MapVIT(pl.LightningModule):
    def __init__(self, vit_output_dim=1000, ravqa_embedding_dim=768):#vit_output_dim=1000
        super(MapVIT, self).__init__()
        config = CONFIGS["ViT-B_16-224"]
        self.vit = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        self.vit.load_from(np.load('ViT-B_16-224.npz'))
        self.vit.eval()
        self.map = nn.Linear(vit_output_dim, ravqa_embedding_dim)


    def forward(self, x):
        x = self.vit(x)
        return self.map(x)

    # Optionally, add training, validation, test steps, etc.
''' Initialize the mapping layer
mapping_layer = MappingLayer(vit_output_dim=1024, ravqa_embedding_dim=your_ravqa_embedding_dim)


mapped_image_features = mapping_layer(image_features)'''
