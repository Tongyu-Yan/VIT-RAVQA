from PIL import Image
import requests
from models.mapping_layers.Mp_layer import MapVIT
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPVisionConfig
from easydict import EasyDict 
import json
import _jsonnet

# Path to your JSONNET file
jsonnet_file_path = '../configs/okvqa/DPR_v.jsonnet'

# Parse the JSONNET file
parsed_json = _jsonnet.evaluate_file(jsonnet_file_path)

# Convert the parsed JSONNET (which is now standard JSON) to a Python dict
config_dict = json.loads(parsed_json)

# Print the configuration

config = EasyDict(config_dict)
model = MapVIT(config)
#model = model.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)


inputs = processor(images=image, return_tensors="pt", padding=True)
#inputs = inputs.pixel_values
print(inputs.pixel_values.shape)

outputs = model(inputs.pixel_values)
print(outputs.last_hidden_state[:,0].shape)
print(outputs.pooler_output.shape)
