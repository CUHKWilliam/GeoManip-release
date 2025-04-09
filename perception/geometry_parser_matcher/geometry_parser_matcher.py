from utils.registry import PERCEPTION
from .models.DCAMA import DCAMA
from .models.utils import to_device
import torch
import os
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

@PERCEPTION.register_module()
class GeometryParserMatcher:
    def __init__(self, config):
        self.config = config
        self.matcher = DCAMA(config['backbone'], 
                             config['feature_extractor_path'], 
                             True, 
                             True, 
                             True,
                             device=config['device'])
        self.matcher.eval()
        self.matcher.to(config['device'])

        self.matcher = nn.DataParallel(self.matcher)

        params = self.matcher.state_dict()
        state_dict = torch.load(config['matcher_checkpoint_path'], map_location=self.config['device'])

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        state_dict2 = {}
        for k, v in state_dict.items():
            if 'scorer' not in k:
                state_dict2[k] = v
        state_dict = state_dict2

        for k1, k2 in zip(list(state_dict.keys()), params.keys()):
            state_dict[k2] = state_dict.pop(k1)

        self.matcher.load_state_dict(state_dict, strict=False)

        database = self.build_database(config['database_root'])
        self.database = database
        img_size = config['img_size']
        img_mean = config['img_mean']
        img_std = config['img_std']

        self.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(img_mean, img_std)])
        

    def build_database(self, database_root):
        database = {}
        for dir in os.listdir(database_root):
            if os.path.isdir(os.path.join(database_root, dir)):
                database[dir] = os.listdir(os.path.join(database_root, dir))
        return database
    
    ## TODO: Very very naively implementations for now
    def retrieve_desc(self, geometry_description):
        descs = list(self.database.keys())
        keyword_matched = -1
        matched_index = -1
        for i, desc in enumerate(descs):
            words = desc.split(" ")
            cnt = 0
            for word in words:
                if word in geometry_description:
                    cnt += 1
            if cnt > keyword_matched:
                keyword_matched = cnt
                matched_index = i
        return descs[matched_index]

    ## TODO: Very very naively implementation for now.
    def retrieve_images(self, geometry_description):
        # step 1: retrieve the most similar description
        retrieved_desc = self.retrieve_desc(geometry_description)
        # step 2: get images
        image_names = self.database[retrieved_desc]
        images = []
        masks = []
        for image_name in image_names:
            if "mask" in image_name:
                continue
            image_path = os.path.join(self.config['database_root'], retrieved_desc, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue
            images.append(image)
            mask_path = os.path.join(self.config['database_root'], retrieved_desc, image_name + "_mask.png")
            mask = cv2.imread(mask_path)
            masks.append(mask)
        return images, masks

    def try_parse(self, image_path, geometry):
        return self.parse(image_path, geometry)
    def parse(self, image_path, geometry):
        support_images, support_masks = self.retrieve_images(geometry)
        nshot = len(support_images)
        image = Image.open(image_path).convert("RGB")
        org_img_size = [image.size[0], image.size[1]]
        query_img = self.transform(image)
   
        support_imgs = torch.stack([self.transform(Image.fromarray(support_img)) for support_img in support_images])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(torch.from_numpy(smask[:, :, 0]).to(self.config['device']).unsqueeze(0).unsqueeze(0).float(), support_imgs.shape[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)
        batch = {
            'support_imgs':support_imgs.to(self.config['device']).unsqueeze(0),
            'query_img': query_img.unsqueeze(0).to(self.config['device']),
            "support_masks": (support_masks > 0).float().to(self.config['device']).unsqueeze(0),
            "org_query_imsize": org_img_size
        }
        pred_mask, simi, simi_map = self.matcher.module.predict_mask_nshot(batch, nshot=nshot)
        return pred_mask[0].cpu().numpy()

if __name__ == '__main__':
    config = {
        'backbone': 'swin',
        'feature_extractor_path': '../../saved_pretrained/swin_base_patch4_window12_384_22kto1k.pth',
        'matcher_checkpoint_path': '../../saved_pretrained/matcher_swin.pt',
        "device": "mps",
        "database_root": "../../database",
        "img_mean": [0.485, 0.456, 0.406],
        "img_std": [0.229, 0.224, 0.225],
        "img_size": 384
    }
    matcher = GeometryParserMatcher(config)
    mask = matcher.parse("test.png", "the handle of the cup")
    rgb = cv2.imread("test.png")
    rgb[mask > 0] = [0, 0, 255]
    cv2.imwrite('test_mask.png', rgb)
