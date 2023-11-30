import pytorch_lightning as pl
import torch.nn as nn
from urllib.request import urlretrieve
import numpy as np
import os
import sys
sys.path.append('/home/ty308/rds/hpc-work/myvqa/Tony-VQA/src/models/VIT')

from modeling import VisionTransformer, CONFIGS

# Define the directory and file path
checkpoint_dir = '/home/ty308/rds/hpc_work/myvqa'
checkpoint_file = os.path.join(checkpoint_dir, "ViT-B_16-224.npz")

# Create the directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Download the file if it does not exist
if not os.path.isfile(checkpoint_file):
    print(f"Downloading ViT-B_16-224.npz to {checkpoint_file}")
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", checkpoint_file)
else:
    print(f"File already exists at {checkpoint_file}")

# Define the MapVIT class
class MapVIT(pl.LightningModule):
    def __init__(self, vit_output_dim=1000, ravqa_embedding_dim=768):
        super(MapVIT, self).__init__()
        config = CONFIGS["ViT-B_16"]
        self.vit = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        
        # Load the file with the absolute path
        self.vit.load_from(np.load(checkpoint_file))
        self.vit.eval()
        self.map = nn.Linear(vit_output_dim, ravqa_embedding_dim)

    def forward(self, x):
        x = self.vit(x)
        return self.map(x)

# Rest of your code...
