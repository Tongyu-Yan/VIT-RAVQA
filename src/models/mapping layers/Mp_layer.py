import pytorch_lightning as pl
import torch.nn as nn

class MappingLayer(pl.LightningModule):
    def __init__(self, vit_output_dim, ravqa_embedding_dim):
        super(MappingLayer, self).__init__()
        self.linear = nn.Linear(vit_output_dim, ravqa_embedding_dim)

    def forward(self, x):
        return self.linear(x)

    # Optionally, add training, validation, test steps, etc.
''' Initialize the mapping layer
mapping_layer = MappingLayer(vit_output_dim=1024, ravqa_embedding_dim=your_ravqa_embedding_dim)


mapped_image_features = mapping_layer(image_features)'''
