from utils.registry import PERCEPTION
import cv2
import torch
from PIL import Image
import numpy as np

@PERCEPTION.register_module()
class NaiveCropper:
    def __init__(self, config):
        self.config = config
    def crop_obj(self, img_path, obj_name,):
        image_source = cv2.imread(img_path)
        obj_image = image_source.copy()
        bbox = self.config['bbox']
        obj_image = obj_image[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
        return obj_image, bbox

@PERCEPTION.register_module()
class GroundingDinoCropper:
    def __init__(self, config):
        self.grounding_dino_model = load_model(config['grounding_dino_config_path'], config['grounding_dino_weight_path'])        
        self.config = config

    def crop_obj(self, img_path, obj_name, ):
        """
        Crop object bounding box using GroundingDino
        """
        image_source = cv2.imread(img_path)
        ## TODO: grounding dino can only take the nounce as input. Using 'a', 'the' degrades its performance severely.
        obj_name = obj_name.replace("the", "").replace("a ", "").strip()

        inputs = self.processor(images=Image.fromarray(image_source), text=obj_name)
        for key in inputs:
            inputs[key] = inputs[key].to(self.config['device'])
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )[0]
        box = results['boxes'][0]
        w, h = image_source.shape[1], image_source.shape[0]
        box *= torch.tensor([w, h, w, h]).to(self.config['device'])
        
        h, w = image_source.shape[0], image_source.shape[1]
        obj_image = image_source[max(int(box[1]) - self.margin, 0): min(int(box[3]) + self.margin, h - 1), max(int(box[0]) - self.margin, 0): min(int(box[2]) + self.margin, w - 1), :]
        return obj_image, box
    