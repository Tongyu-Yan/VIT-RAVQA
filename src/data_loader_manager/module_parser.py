from ast import Raise
from typing import Optional
from easydict import EasyDict
import torch
from transformers import CLIPProcessor
# ! change this file
# 1. input module def imageinput, with a field image
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
class ModuleParser():
    """
    This is a module inherited by the dataset class
    This class is used to parse the sample to form input/output/decoder_input data
    It should be able to process both text-based features and image-based features
    Process:
        (1) Sample-level Sub Parsers:
            Add data fields to the sample
        (2) Porcessing: 
            Aggregating features from individual sub parsers
            Strings under the same field will be automatically concatenated
            Use different fields for different image-based features
        (3) Post-processing:
            Add post-processing units to process the data after parsing
            e.g. tokenization, adding new sample-level features
    """
    def __init__(self) -> None:
        pass
    
    def ImageInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Parse the image input
        """
        return_dict = EasyDict(
            image=None,
        )
        image = sample.image
        inputs = processor(images=image, return_tensors="pt", padding=True)
        if module.option == 'default':
            return_dict.image = inputs.pixel_values
        return return_dict


    def QuestionInput(self, sample: EasyDict, module: EasyDict) -> EasyDict:
        """
        Parse the question input
        Simple add the question to the text sequence
        """
        
        return_dict = EasyDict(
            text_sequence="",
        )
        if module.option == 'default':
            input_sequence = ' '.join([module.separation_tokens.start] + [sample.question] + [module.separation_tokens.end])
        
        return_dict.text_sequence = input_sequence
        return return_dict

    def TextBasedVisionInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Default TextBasedVisionInput module parser
        object: text-based objects, with attributes and OCR'ed texts
        caption: iamge captions
        """
        return_dict = EasyDict(
            text_sequence="",
        )

        # Input from Vision
        vision_sentences = []
        if module.option == 'object':
            vision_sentences += [module.separation_tokens.start]
            for obj in sample.objects:
                attribute_max = module.get('attribute_max', 0)
                if attribute_max > 0:
                    # find suitable attributes
                    suitable_attributes = []
                    for attribute, att_score in zip(obj['attributes'], obj['attribute_scores']):
                        if att_score > module.attribute_thres and len(suitable_attributes) < attribute_max:
                            suitable_attributes.append(attribute)
                    # append to the sentence
                    vision_sentences += suitable_attributes
                vision_sentences.append(obj['class'])
                vision_sentences.append(module.separation_tokens.sep)
            
            ocr = module.get('ocr', 0)
            if ocr > 0:
                text_annotations = sample.img_ocr
                filtered_descriptions = []
                for text_annoation in text_annotations:
                    description = text_annoation['description'].strip()
                    description = description.replace('\n', " ") # remove line switching
                    # vision_sentences += [description]
                    # print('OCR feature:', description)
                    if description not in filtered_descriptions:
                        filtered_descriptions.append(description)
                # print('OCR feature:', filtered_descriptions)
                vision_sentences += filtered_descriptions

            vision_sentences += [module.separation_tokens.end]
            return_dict.text_sequence = ' '.join(vision_sentences)
        
        elif module.option == 'caption':
            return_dict.text_sequence = ' '.join([module.separation_tokens.start] + [sample.img_caption['caption']] + [module.separation_tokens.end])
            
        return return_dict


    def GenerationOutput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Parse the default generation output from gold_answer
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        output_sequence = sample.gold_answer
        return_dict.text_sequence = output_sequence
        return return_dict
    
    def KnowledgeInput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Parse the knowledge input
        """
        return_dict = EasyDict(
            text_sequence="",
        )
        return_dict.text_sequence = ' '.join([module.separation_tokens.start] + [sample.passage_content] + [module.separation_tokens.end])
        return return_dict

    def SimilarityOutput(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Generate the similarity output
        """
        label = [1]
        label += [0] * len(sample.neg_passage_ids)
        return_dict = EasyDict(
            label=label,
        )
        return return_dict


    def parse_modules(self,
                    sample: EasyDict, 
                    modules: EasyDict, 
                    type: str,
                    process_modules: Optional[EasyDict] = None,
                    ) -> Optional[EasyDict]:
        """
        Parse the sample to form input/output/decoder_input
        Args:
            sample: sample to be parsed
            modules: modules to be parsed
            type: type of the module
        Returns:
            parsed sample
        """
        data_collection = []
        if type == "input":
            for input_module in modules:
                parser_func = getattr(self, input_module.type)
                parsed_data = parser_func(sample, input_module)
                data_collection.append(parsed_data)
        elif type == "decoder_input":
            for input_module in modules:
                parser_func = getattr(self, input_module.type)
                parsed_data = parser_func(sample, input_module)
                data_collection.append(parsed_data)
        elif type == "output":
            for output_module in modules:
                parser_func = getattr(self, output_module.type)
                parsed_data = parser_func(sample, output_module)
                data_collection.append(parsed_data)
        else:
            raise ValueError("Unknown type: {}".format(type))
        
        # Process the sample data after aggregating from individual sub parsers
        # before returning to colln_func
        processed_data = data_collection
        if process_modules is None:
            # Run default processing unit
            processed_data = self.DefaultProcessing(processed_data)
        else:
            # Run provided processing unit
            for process_module in process_modules:
                process_func = getattr(self, process_module.type)
                processed_data = process_func(processed_data)

        return processed_data

    def DefaultProcessing(self, data_to_process: EasyDict) -> EasyDict:
        """
        Process the sample data after aggregating from individual sub parsers
        """
        processed_data = EasyDict()
        for data_entry in data_to_process:
            for key, value in data_entry.items():
                if key not in processed_data:
                    processed_data[key] = value
                else:
                    if type(value) == str:
                        # automatically concatenate strings with the same key
                        processed_data[key] += ' ' + value
                    else:
                        raise TypeError("Undefined processing type: {}".format(type(value)))
        
        return processed_data



    def PostProcessInputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for input tokenization
        """
        assert 'text_sequence' in data_to_process.keys()
        text_sequences = data_to_process.pop('text_sequence')
        task_prefix = ""
        encoding = self.tokenizer([task_prefix + sequence for sequence in text_sequences],
                            padding='longest',
                            max_length=self.config.data_loader.additional.max_source_length,
                            truncation=True,
                            return_tensors="pt")
        data_to_process.update({
            'input_ids': encoding.input_ids,
            'attention_mask': encoding.attention_mask,
            'input_text_sequences': text_sequences,
        })
        return data_to_process
    
    def PostProcessDecoderInputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for decoder input tokenization
        """
        assert 'text_sequence' in data_to_process.keys()
        text_sequences = data_to_process.pop('text_sequence')
        encoding = self.decoder_tokenizer([sequence for sequence in text_sequences],
                            padding='longest',
                            max_length=self.config.data_loader.additional.max_decoder_source_length,
                            truncation=True,
                            return_tensors="pt")
        data_to_process.update({
            'decoder_input_ids': encoding.input_ids,
            'decoder_input_attention_mask': encoding.attention_mask,
            'decoder_input_text_sequences': text_sequences,
        })
        return data_to_process

    def PostProcessOutputTokenization(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for output tokenization
        """
        assert 'text_sequence' in data_to_process.keys()
        text_sequences = data_to_process.pop('text_sequence')
        target_encoding = self.decoder_tokenizer(text_sequences,
                                    padding='longest',
                                    max_length=self.config.data_loader.additional.max_target_length,
                                    truncation=True)
        labels = target_encoding.input_ids
        output_sequence_ids = target_encoding.input_ids # For teacher force training
        output_sequence_ids = torch.LongTensor(output_sequence_ids)
        output_sequence_attention_mask = torch.LongTensor(target_encoding.attention_mask) # For teacher force training
        
        # replace padding token id's of the labels by -100
        labels = [
            [(label if label != self.decoder_tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
        ]

        labels = torch.LongTensor(labels)
        assert labels.shape == output_sequence_ids.shape

        data_to_process.update({
            'labels': labels,
            'output_sequence_ids': output_sequence_ids,
            'output_sequence_attention_mask': output_sequence_attention_mask,
            'output_text_sequences': text_sequences,
        })
        return data_to_process
    # ! add image processing to the below two functions
    def PostProcessConcatenateLabels(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for concatenating labels
        """
        assert 'label' in data_to_process.keys()
        label = data_to_process.pop('label')
        labels = []
        for l in label:
            labels += l
        data_to_process.update({
            'labels': torch.LongTensor(labels),
        })
        return data_to_process

    def PostProcessImage(self, data_to_process: EasyDict) -> EasyDict:
        """
        Post-processing for images
        """
        assert 'image' in data_to_process.keys()
        image = data_to_process.pop('image')
        data_to_process.update({
            'images': image,
        })
        return data_to_process

    def post_processing(self, 
                        processed_batch_data: EasyDict,
                        postprocess_modules: Optional[EasyDict] = None,) -> EasyDict:
        """
        Post-processing the processed data of the whole batch
        Called by colln_func after processing each sample
        """
        postprocessed_batch_data = processed_batch_data
        if postprocess_modules is None:
            # Do nothing and return
            return postprocessed_batch_data
        else:
            # Run provided post-processing unit
            for postprocess_module in postprocess_modules:
                process_func = getattr(self, postprocess_module.type)
                postprocessed_batch_data = process_func(postprocessed_batch_data)
        
        return postprocessed_batch_data