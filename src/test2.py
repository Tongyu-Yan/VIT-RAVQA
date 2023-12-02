import json
import _jsonnet
from easydict import EasyDict

#jsonnet_file_path = '/home/ty308/rds/hpc-work/myvqa/Tony-VQA/configs/okvqa/DPR_v.jsonnet'
jsonnet_file_path = '../configs/okvqa/DPR_v.jsonnet'

# Parse the JSONNET file
parsed_json = _jsonnet.evaluate_file(jsonnet_file_path)

# Convert the parsed JSONNET (which is now standard JSON) to a Python dict
config_dict = json.loads(parsed_json)
config = EasyDict(config_dict)
toprint = config.vit_model_config.VisionModelConfigClass

# Print the configuration
print(json.dumps(toprint, indent=4))