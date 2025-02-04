import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
from torchvision.ops import box_convert
import base64
from openai import OpenAI
from utils import exec_safe
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from utils import *
from skimage.measure import label as Label
import cv2


SEGM_PROMPT1 = '''
Here are some knowledge about finding the parts given segmentation masks: {}
knowledge end.
There are totally {} pair of images. 
For each pair, the left image is the image of {} with different part highlighted in red. The right image is the segmentation mask highlighted in white to represent different parts of {}. These images are named as image i, ... (i=0, 1, 2, ...)
    Please infer what is highlighted in red for the left image one by one, and then select one of the image of {}.
    - Output: image i, `geometry` (i=0,1,2... is the index number) at the end in a single line.
    - Where `geometry` is the geometry of object, like the edge, the center, the area, left point, right, point, etc..
    - If the segmentation image does not contain the object part, think about whether we can derive the object part from this image, and select this image. For example, if the image does not correspond to "the tip of the pen", output the mask containing the pen and we can derive the tip later.
    - You can analysis the problem if needed, but please output the final result in a seperate line in the format image i, `part`.
    - For the right image, check if the corresponding object part is in black. If so, it is a background and don't use it !!!!!!!!!
    - Remember that the image index i starts from 0.
    - At the end, output "<splitter>"
    '''

SEGM_PROMPT2 = '''
    Here are some knowledge about finding the parts given segmentation masks: {}
    Write a Python function to find out the {} given the segmentation of image {}, {}. 
    - the input `mask` is a boolean numpy array of a segmentation mask in shapes (H, W)
    - return the mask which is a numpy array. 
    - You can `import numpy as np` and `import cv2`, but don't import other packages
    - mask_output should still be in the shape(H, W)
    ## code start here
    def segment_object(mask):
        ...
        return mask_output
    Please directly output the code without explanations. Complete the comment in the code. Remove import lines since they will be manually imported later.'''

def process_point_cloud(pc):
    pc = pc[np.logical_not(np.isnan(pc).any(-1))]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pc = np.asarray(pcd.remove_radius_outlier(nb_points=5, radius=0.02)[0].points)
    return pc

def get_point_cloud(obj_description, timestamp=-1):
    import utils as utils
    if utils.ENV.part_to_pts_dict_simulation is not None:
        part_to_pts_dict = utils.ENV.part_to_pts_dict_simulation.copy()
    else:
        part_to_pts_dict = utils.ENV.part_to_pts_dict.copy()
    if abs(timestamp) > len(part_to_pts_dict):
        pc = part_to_pts_dict[-1][obj_description]
    else:
        pc = part_to_pts_dict[timestamp][obj_description]
    return pc

gvars_dict = {
    'np': np,
    'cv2': cv2,
}
class GeometryParser:
    def __init__(self, config):
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.margin_ratio = config['margin_ratio']
        self.box_threshold = config['box_threshold']
        self.text_threshold = config['text_threshold']
        self.mask_ratio_filter = config['mask_ratio_filter'] if "mask_ratio_filter" in config.keys() else 0.01
        self.erode_ratio = config['erode_ratio'] if "erode_ratio" in config.keys() else 0.1
        self.vlm_model = config['model'] if "model" in config.keys() else "chatgpt-4o-latest"
        self.vlm_model_role = "system" if "o1" not in self.vlm_model else "assistant"
        root_path = os.path.dirname(os.path.realpath(__file__))
        sam = sam_model_registry["vit_h"](os.path.join(root_path, "sam_vit_h_4b8939.pth")).cuda()
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
        )

        self.groundingdino_model = load_model(os.path.join(root_path, "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"), os.path.join(root_path, "../GroundingDINO/weights/groundingdino_swint_ogc.pth")).cuda()

    def parse(self, obj_description, timestamp=-1, image_path=None, rekep_program_dir=None, seed=0,):
        image_source, image = load_image(image_path)

        ## this is for user provided gt mask 
        if rekep_program_dir is not None and os.path.exists(os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, "gt")):
            mask_gt_path = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, "gt")
            mask_gt = cv2.imread(mask_gt_path)
            box = [0, 0, image_source.shape[1] - 1, image_source.shape[0] - 1]
            margin = 0
            w, h = image_source.shape[1], image_source.shape[0]
            masks = mask_gt[None, ...]
        else:
            obj_name = obj_description.split(" of ")[-1].strip()
            boxes, logits, phrases = predict(
                model=self.groundingdino_model,
                image=image,
                caption=obj_name.replace("the", "").strip(),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            w, h = image_source.shape[1], image_source.shape[0]
            boxes *= torch.tensor([[w, h, w, h]]).to(image.device)
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            ## pad the image a little bit for the SAM to work well
            if len(boxes) == 0:
                return np.zeros((image_source.shape[0], image_source.shape[1]))
            box = boxes[0]

            ## TODO: for shelf-place task on "corner2" and "corner3" view in the MetaWorld exp, choose box 1 for identify the shelf since GroundingDINO think the table is the shelf...
            # if "shelf" in obj_description:
            #     box = boxes[1]
            # else:
            #     box = boxes[0]
            ## TODO: end 

            margin = int(self.margin_ratio * min((box[3] - box[1]), (box[2] - box[0])))
            obj_image = image_source[max(int(box[1]) - margin, 0): min(int(box[3]) + margin, h - 1), max(int(box[0]) - margin, 0): min(int(box[2]) + margin, w - 1), :]
            cv2.imwrite(os.path.join(rekep_program_dir, "object_{}.png".format(obj_description)), obj_image[:, :, ::-1])

            masks_dict = self.mask_generator.generate(obj_image)
            masks = np.stack([mask_dict['segmentation'] for mask_dict in masks_dict], axis=0)

            masks2 = []
            ## mask filtering
            for mask in masks:
                if (mask > 0).sum().astype(np.float32) / (mask > -1).sum()  > self.mask_ratio_filter and (mask > 0).sum() > 10:
                    masks2.append(mask)
            masks2 = np.stack(masks2, axis=0)
            masks = masks2
            masks2 = []
            mask_path0 = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, 0)
            if not os.path.exists(mask_path0):
                for idx in range(len(masks)):
                    mask_path = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, idx)
                    mask2 = np.hstack((obj_image.copy() * 1 + np.repeat(((masks[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1) * np.array([255, 0, 0]) * 0.5, np.repeat(((masks[idx].copy() > 0) * 255)[:, :, None], repeats=3, axis=-1)))
                    cv2.imwrite(mask_path, mask2[:, :, ::-1])
                    masks2.append(mask2)
            else:
                for idx in range(50):
                    mask_path = os.path.join(rekep_program_dir, "mask_{}_{}.png").format(obj_description, idx)
                    if not os.path.exists(mask_path):
                        break
                    mask2 = cv2.imread(mask_path)
                    masks2.append(mask2)
            masks2 = np.stack(masks2, axis=0)
            masks = masks2
        if rekep_program_dir is not None:
            cache_segment_program_path = os.path.join(rekep_program_dir, "program_segm_{}.txt".format("-".join(obj_description.split(" "))))
        else:
            cache_segment_program_path = None
        if rekep_program_dir is None or not os.path.exists(cache_segment_program_path):
            contents = []
            ## TODO: use prompt
            segm_prompt_root = "./segm_prompts"
            prompt_dirs = os.listdir(segm_prompt_root)
            for prompt_dir in prompt_dirs:
                prompt_raw_path = os.path.join(segm_prompt_root, prompt_dir, "prompt.txt")
                with open(prompt_raw_path, "r") as f:
                    prompt_raw = f.read()
                contents += parse_prompt(os.path.join(segm_prompt_root, prompt_dir), prompt_raw)
            with open("./vlm_query/part_knowledge.txt", "r") as f:
                part_knowdge = f.read()
        
            client = OpenAI()
            base64_image = encode_image(image_path)

            messages = [{"role": self.vlm_model_role, "content": "You are a helpful assistant."}]
            
            for idx in range(len(masks)):
                base64_image = encode_image(os.path.join(rekep_program_dir, "mask_{}_{}.png".format(obj_description, idx)))
                contents.append(
                    {
                    "type": "text",
                        "text": "The next image is the image {}.".format(idx)
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
                            "text": "What is highlighted in red ?"
                        }
                    )
            contents.append(
            {
                    "type": "text",
                    "text": SEGM_PROMPT1.format(part_knowdge, len(masks), obj_name, obj_name, obj_description),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": contents
                }
            )

            reply1 = query_vlm_model(client, self.vlm_model, messages, self.temperature, self.top_p)
            mask_indice = int(reply1.lower().split("image ")[-1].split(",")[0].strip())
            part = reply1.split("image ")[-1].split(",")[-1].split("\n")[0].replace("*", "").strip()
            segm_prompt2 = SEGM_PROMPT2.format(part_knowdge, obj_description, mask_indice, part, part, obj_description)
            messages.append({
                "role": self.vlm_model_role, "content": reply1
            })
            messages.append({
            "role": "user",
            "content":[{ 
                    "type": "text",
                    "text": segm_prompt2}]
            })
            reply = query_vlm_model(client, self.vlm_model, messages, self.temperature, self.top_p)
            code = reply.split("python\n")[1].replace("import numpy as np", "").replace("```", "")
            code = "## mask_indice: {}\n".format(mask_indice) + code
            ## TODO:
            # import ipdb;ipdb.set_trace()
            if cache_segment_program_path is not None:
                with open(cache_segment_program_path, "w") as f:
                    f.write(code)
        else:
            with open(cache_segment_program_path, "r") as f:
                code = f.read()
            mask_indice = code.split("\n")[0].split(":")[1].strip()
            if mask_indice.strip() != "gt":
                mask_indice = int(code.split("\n")[0].split(":")[1])
            
        lvars = {}
        exec_safe(code, gvars_dict, lvars)
        if mask_indice != "gt":
            mask = masks[mask_indice]
        else:
            mask = masks[0]
        H, W = mask.shape[0], mask.shape[1]
        mask = mask[:H, W//2:, 0] > 0
        if mask[0, 0] and mask[0, W // 2 - 1] and mask[H - 1, 0] and mask[H - 1, W // 2 - 1] and np.logical_not(mask).sum() > 10:
            mask = np.logical_not(mask)
        ## TODO: erode the mask a little bit to prevent projecting to the background
        erode_amount = (int(np.sqrt(mask.sum()) * self.erode_ratio))
        if erode_amount > 0:
            eroded_mask = cv2.erode(mask.astype(np.uint8), np.ones((erode_amount, erode_amount), np.uint8))
            mask = eroded_mask > 0
        segm = lvars['segment_object'](mask)
        labels = Label(segm)
        try:
            segm = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        except:
            import ipdb;ipdb.set_trace()
        segm = (segm > 0) * 255
        
        segm2 = np.zeros((image_source.shape[0], image_source.shape[1]))
        if segm.shape[0] != min(int(box[3]) + margin, h - 1) - max(int(box[1]) - margin, 0) or segm.shape[1] != min(int(box[2]) + margin, w - 1) -  max(int(box[0]) - margin, 0):
            segm = cv2.resize(segm.astype(np.uint8).copy(), (min(int(box[2]) + margin, w - 1) -  max(int(box[0]) - margin, 0), min(int(box[3]) + margin, h - 1) - max(int(box[1]) - margin, 0) ))
        segm2[max(int(box[1]) - margin, 0): min(int(box[3]) + margin, h - 1), max(int(box[0]) - margin, 0): min(int(box[2]) + margin, w - 1)] = segm
        ## TODO: for debug
        cv2.imwrite(
            "debug2.png", (segm2 > 0).astype(np.uint8) * 255)
        return segm2

    def query_segment(self, segm_mask, part_description):
        if part_description.strip() == "":
            code = "def segment_object(mask): return mask"
            return segm_mask, code
        messages = [{"role": self.vlm_model_role, "content": "You are a helpful assistant."}]
        contents = []
        with open("./vlm_query/part_knowledge.txt", "r") as f:
            part_knowdge = f.read()
        prompt = SEGM_PROMPT2.format(part_knowdge, part_description, "", "", part_description)
        segm_tmp_dir = "./tmp"
        segm_tmp_path = os.path.join(segm_tmp_dir, "{}.png".format("_".join(part_description.split("\n"))))
        cv2.imwrite(segm_tmp_path, segm_mask.astype(np.uint8) * 255)
        base64_image = encode_image(segm_tmp_path)
        contents.append(
                {
                "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
        )
        contents.append(
            {
            "type": "text",
                "text": prompt
            }
        )
        messages.append(
            {
                "role": "user",
                "content": contents
            }
        )
        client = OpenAI()
        reply = query_vlm_model(client, self.vlm_model, messages, self.temperature, self.top_p)
        code = reply.split("python\n")[1].replace("import numpy as np", "").replace("```", "")
        lvars = {}
        exec_safe(code, gvars_dict, lvars)
        ## TODO: erode the mask a little bit to prevent projecting to the background
        erode_amount = (int(np.sqrt(segm_mask.sum()) * self.erode_ratio))
        eroded_mask = cv2.erode(segm_mask.astype(np.uint8), np.ones((erode_amount, erode_amount), np.uint8))
        segm_mask = eroded_mask > 0
        try:
            segm = lvars['segment_object'](segm_mask)
        except:
            import ipdb;ipdb.set_trace()
        labels = Label(segm)
        segm = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        segm = segm > 0
        return segm, code

if __name__ == "__main__":
    config = {
        "temperature": 0.1,
        "top_p": 0.1,
        "margin_ratio": 0.1,
        "box_threshold": 0.3,
        "text_threshold": 0.2,
    }
    segmentor = Segmentor(config=config)
    mask = segmentor.segment("the body of the cup", image_path="./cup.png", rekep_program_dir="./tmp")
    mask = np.repeat(mask[:, :, None], repeats=3, axis=-1)
    mask[:, :, 1:] = 0
    rgb = cv2.imread("./pour_macaroni.png")
    rgb[mask[:, :, 0] > 0] = np.array([0, 0, 255])
    cv2.imwrite('debug.png', rgb)
    import ipdb;ipdb.set_trace()