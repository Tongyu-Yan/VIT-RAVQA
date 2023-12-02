from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPVisionConfig
config = CLIPVisionConfig()
model = CLIPVisionModel(config)
model = model.from_pretrained("openai/clip-vit-base-patch32")
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
