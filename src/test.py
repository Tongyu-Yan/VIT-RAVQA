from data_loader_manager.data_loader_okvqa import DataLoaderOKVQA
from utils.config_system import get_config_from_json, process_config
import argparse
from easydict import EasyDict


jsonfile = '/home/ty308/rds/hpc-work/myvqa/Tony-VQA/configs/okvqa/DPR.jsonnet'
config, config_dict = get_config_from_json(jsonfile)
args = argparse.Namespace(config=jsonfile, reset='your_value', mode='your_value', 
                           experiment_name='your_value', modules=[], tags=[], 
                           test_batch_size=-1, test_evaluation_name='', opts=[])

# Now pass this mock namespace to process_config
config = process_config(args)


dataloader = DataLoaderOKVQA(config)
dataloader.build_dataset()

dataset_modules = config.data_loader.dataset_modules.module_list
module_config = config.data_loader.dataset_modules.module_dict.LoadOKVQAData
#print("Module Config for", dataset_module, ":", module_config)
img = dataloader.LoadOKVQAData(module_config)
print(img)