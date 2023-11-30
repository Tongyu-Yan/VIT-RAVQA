import pytorch_lightning as pl
import torch.nn as nn
import sys
sys.path.append('/home/ty308/rds/hpc-work/myvqa/Tony-VQA/src/models/VIT')

from modeling import VisionTransformer

import os

checkpoint_dir = home/ty308/rds/hpc_work/myvqa
os.makedirs(checkpoint_dir, exist_ok=True)
if not os.path.isfile("home/ty308/rds/hpc_work/myvqa/ViT-B_16-224.npz"):
    urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "attention_data/ViT-B_16-224.npz")

class MapVIT(pl.LightningModule):
    def __init__(self, vit_output_dim=1000, ravqa_embedding_dim=768):#vit_output_dim=1000
        super(MappingLayer, self).__init__()
        self.vit = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
        self.vit.load_from(np.load('ViT-B_16.npz'))
        self.vit.eval()
        self.map = nn.Linear(vit_output_dim, ravqa_embedding_dim)


    def forward(self, x):
        x = self.vit(x)
        return self.map(x)

    # Optionally, add training, validation, test steps, etc.
''' Initialize the mapping layer
mapping_layer = MappingLayer(vit_output_dim=1024, ravqa_embedding_dim=your_ravqa_embedding_dim)


mapped_image_features = mapping_layer(image_features)'''
