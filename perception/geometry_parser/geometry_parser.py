## TODO: Not done !!!!!
import numpy as np
import cv2
import torch
from torchvision.ops import box_convert
import base64
from openai import OpenAI
from utils.utils import exec_safe
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import scipy
import re
import cv2
from utils.registry import PERCEPTION
from ..perception_base import PerceptionBase
import cv2

@DeprecationWarning
@PERCEPTION.register_module()
class GeometryParser(PerceptionBase):
    def __init__(self, config):
        self.box_threshold = config['box_threshold']
        self.text_threshold = config['text_threshold']
        self.filter_ratio = config['filter_ratio']
        self.margin = config['margin']
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.config = config
        self.task_dir = config['task_dir']

        self.sam = sam_model_registry["vit_h"](config['sam_weight_path'])  # Use vit_h, vit_l, or vit_b based on the model
        self.sam.to(self.config['device'])
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
        )

        ## Qwen
        self.client = OpenAI(api_key=config['vlm_api_key'], base_url=config['vlm_base_url'])
        self.vlm_model = config['vlm_model']


    def filter_masks(self, masks):
        """
        Filter masks based on a size threshold.

        Args:
            masks (numpy.ndarray): Array of masks to be filtered.

        Returns:
            numpy.ndarray: Filtered array of masks.
        """
        H, W = masks.shape[1], masks.shape[2]
        masks2 = []
        ## mask filtering
        for mask in masks:
            if (mask[:, W // 2:, 0] > 0).sum() > W * H * self.filter_ratio:
                masks2.append(mask)
        masks2 = np.stack(masks2, axis=0)
        return masks2
    
    def generate_mask_candidates(self, img, geometry_description, overwrite=False):
        mask_path0 = os.path.join(self.task_dir, "mask_{}_{}.png").format(geometry_description, 0)
        masks = []
        if not os.path.exists(mask_path0) or overwrite:
            masks_dict = self.mask_generator.generate(img)
            masks_candidates = np.stack([mask_dict['segmentation'] for mask_dict in masks_dict], axis=0)
            for idx in range(len(masks_candidates)):
                mask_path = os.path.join(self.task_dir, "mask_{}_{}.png").format(geometry_description, idx)
                mask = np.hstack((img.copy() + np.repeat(((masks_candidates[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1) * np.array([255, 0, 0]) * 0.5, np.repeat(((masks_candidates[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1)))
                cv2.imwrite(mask_path, mask[:, :, ::-1])
                masks.append(mask)
        else:
            for idx in range(50):
                mask_path = os.path.join(self.task_dir, "mask_{}_{}.png").format(geometry_description, idx)
                if not os.path.exists(mask_path):
                    break
                mask = cv2.imread(mask_path)
                masks.append(mask)
        masks_candidates = np.stack(masks, axis=0)
        return masks_candidates

    def select_mask(self, geometry_description, obj_name, mask_candidates, overwrite=False):
        ## TODO: special case, need a white list in the future
        if ("body" in geometry_description or "table" in geometry_description ) and "edge" not in geometry_description and "corner" not in geometry_description:
            mask_sum = [m.sum() for m in mask_candidates]
            return None, mask_candidates[np.argmax(mask_sum)], None
        cache_selected_indice_path = os.path.join(self.task_dir, f"program_indice_{geometry_description}.txt")
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if not os.path.exists(cache_selected_indice_path) or overwrite:
            contents = []
            geometry_parser_example_prompt = open(os.path.join(self.vlm_query_dir, "geometry_parser_example_prompt.txt"), "r").read()
            geometry_parser_scheme_prompt = open(os.path.join(self.vlm_query_dir, "geometry_parser_scheme_prompt.txt"), "r").read()

            for idx in range(len(mask_candidates)):
                selected_mask_path = f"{self.task_dir}/mask_{geometry_description}_{idx}_filitered.png"
                cv2.imwrite(selected_mask_path, mask_candidates[idx])
                # img = np.asarray(Image.open(os.path.join(self.task_dir, "mask_{}_{}.png".format(geometry_description, idx))))
                img = mask_candidates[idx]
                img = img[:, : img.shape[1]//2, :]
                base64_image = encode_image(img)
                contents.append(
                    {
                    "type": "text",
                        "text": "Below is image {}:".format(idx)
                    }
                )
                contents.append(
                        {
                        "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    )
                contents.append(
                        {
                        "type": "text",
                            "text": "What is highlighted in red ? Is this potentially what we want ?"
                        }
                    )
            contents.append(
            {
                    "type": "text",
                    "text": geometry_parser_scheme_prompt.split("<splitter>")[0].format(geometry_parser_example_prompt, len(mask_candidates), obj_name, geometry_description),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": contents
                }
            )
            reply = query_vlm_model(self.client, self.vlm_model, messages, self.temperature, self.top_p)
            messages.append({
                "role": "system", "content": reply
            })
            mask_indice = int(reply.lower().split("image ")[-1].split(",")[0].strip())
            part = reply.split(",")[-1].replace("\'", "").replace("'", "").replace("`","")
            selected_mask = mask_candidates[mask_indice]
            open(cache_selected_indice_path, "w").write(f"{mask_indice},{part}")
        indice = int(open(cache_selected_indice_path, "r").read().split(",")[0].strip())
        selected_mask = mask_candidates[indice]
        return messages, selected_mask, part
    
    def post_process(self, messages, selected_mask, part, geometry_description, overwrite=True):
        import ipdb;ipdb.set_trace()
        if "edge" not in geometry_description and "corner" not in geometry_description: ## TODO: speed up strategy to prevent frequent query. Use a white list to do it later.
            return selected_mask
        cache_segment_program_path = os.path.join(self.task_dir, f"program_segm_{geometry_description}.txt")
        if not os.path.exists(cache_segment_program_path) or overwrite:
            geometry_parser_scheme_prompt = open(os.path.join(self.vlm_query_dir, "geometry_parser_scheme_prompt.txt"), "r").read()

            segm_prompt = geometry_parser_scheme_prompt.split("<splitter>")[-1].format(geometry_description, geometry_description,)
            messages.append({
            "role": "user",
            "content":[{ 
                    "type": "text",
                    "text": segm_prompt}]
            })
            reply = query_vlm_model(self.client, self.vlm_model, messages, self.temperature, self.top_p)
            code = reply.split("<answer>")[1].split("</answer>")[0]
            code = code.split("python\n")[1].replace("import numpy as np", "").replace("```", "")
            code_list = code.split("\n")
            code_list2 = []
            start = False
            for line in code_list:
                if line.strip().startswith("def") or start:
                    code_list2.append(line)
                    start = True
                if line.strip().startswith("return") and start:
                    break
            code = "\n".join(code_list2)
            with open(cache_segment_program_path, "w") as f:
                f.write(code)
        else:
            code = f.open(cache_segment_program_path, "r").read()
        gvars_dict = {
            'np': np,
            "scipy": scipy,
            "cv2": cv2,
            }
        lvars = {}
        exec_safe(code, gvars_dict, lvars)
        processed_mask = lvars[list(lvars.keys())[0]](selected_mask)
        return processed_mask

    def try_parse(self, img, geometry_description):
        try_cnt = 0
        while True:
            try:
                return self.parse(img, geometry_description)
            except:
                print("fail, retry")
                try_cnt += 1
                if try_cnt == 10:
                    raise Exception("segmentation fail")
                continue
            
    def parse(self, img, geometry_description):
        print(f"parsing {geometry_description}")
        cached_path = f"{self.task_dir}/{geometry_description}.png"
        img = img.copy()
        obj_name = geometry_description.split("of")[-1].strip()
        img_cached_path = os.path.join(self.task_dir, "query_img.png")
        if not os.path.exists(img_cached_path):
            cv2.imwrite(img_cached_path, img[:, :, ::-1])
        obj_img, box = self.cropper.crop_obj(img_cached_path, obj_name)
        mask_candidates = self.generate_mask_candidates(obj_img, geometry_description)
        mask_candidates = self.filter_masks(mask_candidates)
        messages, selected_mask, part = self.select_mask(geometry_description, obj_name, mask_candidates)
        H, W = selected_mask.shape[0], selected_mask.shape[1]
        selected_mask = selected_mask[:H, W//2:, 0] > 0
        ## TODO: erode the mask a little bit to prevent projecting to the background
        try_eroded_mask = cv2.erode(selected_mask.astype(np.uint8), np.ones((2, 2), np.uint8))
        if try_eroded_mask.sum() > 0:
            selected_mask = try_eroded_mask > 0
        processed_mask = self.post_process(messages, selected_mask, part, geometry_description)

        w, h = img.shape[1], img.shape[0]
        final_mask = np.zeros((img.shape[0], img.shape[1]))
        if processed_mask.shape[0] != min(int(box[3]) + self.margin, h - 1) - max(int(box[1]) - self.margin, 0) or processed_mask.shape[1] != min(int(box[2]) + self.margin, w - 1) -  max(int(box[0]) - self.margin, 0):
            processed_mask = cv2.resize(final_mask.astype(np.uint8).copy(), (min(int(box[2]) + self.margin, w - 1) -  max(int(box[0]) - self.margin, 0), min(int(box[3]) + self.margin, h - 1) - max(int(box[1]) - self.margin, 0) ))

        final_mask[max(int(box[1]) - self.margin, 0): min(int(box[3]) + self.margin, h - 1), max(int(box[0]) - self.margin, 0): min(int(box[2]) + self.margin, w - 1)] = processed_mask
        cv2.imwrite(cached_path, (final_mask * 255).astype(np.uint8))
        print(f"finish parsing {geometry_description}")
        return final_mask


if __name__ == "__main__":
    geometry_parser = GeometryParser(task_dir="./vlm_query/debug")
    color_image, depth_image = camera.update_image_depth()
    cv2.imwrite("./vlm_query/debug/query_img.png", color_image[:, :, ::-1])
    mask = geometry_parser.parse(img=cv2.imread("vlm_query/debug/query_img.png"), geometry_description="the cup opening of the cup")
    cv2.imwrite("debug.png", mask.astype(np.uint8) * 255)
