from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
import pprint
config = DPRConfig().to_dict()
print(config.keys())

