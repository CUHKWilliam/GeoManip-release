import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
from datetime import datetime
import re
from utils import query_vlm_model
import subprocess
from utils import load_functions_from_txt

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_and_save_cost_fns(output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        meta_data = {}
        max_stage = -1
        objects_to_segment = []
        for i, line in enumerate(lines):
            if line.strip().startswith("def"):
                start = i
                name = line.split("(")[0].split("def ")[1]
                stage = int(name.split("_")[1])
                if stage > max_stage:
                    max_stage = stage
            if line.strip().startswith("return"):
                end = i
                functions[name] = lines[start:end+1]
            if "get_point_cloud" in line:
                obj = line.split("\"")[1]
                objects_to_segment.append(obj)
            if "grasp(\"" in line:
                obj = line.split("\"")[1]
                if obj.strip() != "":
                    objects_to_segment.append(obj)
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")

class GeometricAndCodeGenerator:
    def __init__(self, config, prompt_template_path=None):
        self.config = config
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query')
        self.vlm_model = config['model'] if 'model' in config.keys() else "chatgpu-4o-latest"
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        if prompt_template_path is None:
            prompt_template_path = os.path.join(self.base_dir, 'prompt_template.txt')
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.query_history = [{"role": "system", "content": "You are a helpful assistant that breaks down task, write geometric constraints, and write Python code for robot manipulation. Please learn from the knowledge as much as possible. Think carefully and reason step by step."}]

    def build_prompt_cost_functions(self, stage_num=None, prompt_text_only=False):
        prompt_cost_function = self.prompt_template.split("<STEP SPLITTER>")[1]
        if stage_num is not None:
            prompt_cost_function = prompt_cost_function.replace("for each stage", "for ONLY the {} stage".format(stage_num))
        with open("./vlm_query/geometry_knowledge.txt", "r") as f:
            geometry_knowledge = f.read()
        prompt_text = prompt_cost_function.format(geometry_knowledge)
        if prompt_text_only:
            return prompt_text
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
            ]
        }
        return message

    def build_prompt_geometry_constraints(self, img, instruction, task_dir, hint="", prompt_text_only=False):
        image_path = os.path.join(task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        prompt_geometry_constraints = self.prompt_template.split("<STEP SPLITTER>")[0]
        with open("./vlm_query/geometry_constraints_prompt.txt", 'r') as f:
            goemetry_constranits_prompt = f.read()
        prompt_text = prompt_geometry_constraints.format(instruction, goemetry_constranits_prompt)
        if prompt_text_only:
            return prompt_text
        img_base64 = encode_image(image_path)
        message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        
        return message
        
    def get_object_to_segment(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        meta_data = {}
        max_stage = -1
        objects_to_segment = []
        for i, line in enumerate(lines):
            if line.strip().startswith("def"):
                start = i
                name = line.split("(")[0].split("def ")[1]
                stage = int(name.split("_")[1])
                if stage > max_stage:
                    max_stage = stage
            if line.strip().startswith("return"):
                end = i
                functions[name] = lines[start:end+1]
            if "get_point_cloud" in line:
                obj = line.split("\"")[1]
                objects_to_segment.append(obj)
            if "grasp(\"" in line:
                obj = line.split("\"")[1]
                if obj.strip() != "":
                    objects_to_segment.append(obj)
        objects_to_segment = list(set(objects_to_segment))
        objects_to_segment += [
            "the gripper of the robot",
            "the gripper approach of the robot",
            "the gripper binormal of the robot",
        ]
        meta_data.update({
            "num_stage": max_stage,
            "object_to_segment": objects_to_segment
        })
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(meta_data, f)
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the constraint idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_constraints.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        print(f"Constraints saved to {save_dir}")


    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def task_decomposition_and_geometric_constraint_generation(self, img, instruction, task_dir=None, hint="", seed=None,):
        if task_dir is None:
            # create a directory for the task
            fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
            self.task_dir = os.path.join(self.base_dir, fname)
            os.makedirs(self.task_dir, exist_ok=True)
        prompt = self.build_prompt_geometry_constraints(img, instruction + ". DETAILS: {}.".format(hint), hint, task_dir)
        self.query_history.append(prompt)
        output_constraint_file = os.path.join(task_dir, "output_constraints.txt")
        self.task_dir = task_dir
        if not os.path.exists(output_constraint_file):
            # stream back the response
            stream = query_vlm_model(self.client, self.vlm_model, self.query_history, self.temperature, self.top_p, stream=True)
            output = ""
            start = time.time()
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            with open(output_constraint_file, "w") as f:
                f.write(output)
        else:
            with open(output_constraint_file, "r") as f:
                output = f.read()
            self.query_history.append(
                {"role": "system", "content": "{}".format(output)}
            )
        return output
    
    def cost_fns_generation(self, fns_dict, task_dir=None):
        prompt = self.build_prompt_cost_functions()
        self.query_history.append(prompt)
        output_geom_consts_file = os.path.join(task_dir, "output_geometric_constraints.txt")

        if not os.path.exists(output_geom_consts_file):
            # stream back the response
            stream = query_vlm_model(self.client, self.vlm_model, self.query_history, self.temperature, self.top_p, stream=True)
            output = ""
            start = time.time()
            for chunk in stream:
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
            self.query_history.append(
                {"role": "system", "content": "{}".format(output)}
            )
            self.query_history.append(output)
            # save raw output
            with open(output_geom_consts_file, 'w') as f:
                f.write(output)
        else:
            with open(output_geom_consts_file, "r") as f:
                output = f.read()
        parse_and_save_cost_fns(output, self.task_dir)
        constraint_fns = dict()
        constraint_fns_code = dict()
        for stage in range(1, self.program_info['num_stage'] + 1):  # stage starts with 1
            stage_dict = dict()
            stage_dict_code = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(task_dir, f'stage_{stage}_{constraint_type}_constraints.txt')
                if not os.path.exists(load_path):
                    func, code = [], []
                else:
                    ret = load_functions_from_txt(load_path, fns_dict, return_code=True) 
                    func, code = ret['func'], ret["code"]
                ## merge the target constraints and the sub-goal constraint
                stage_dict[constraint_type] = func
                stage_dict_code[constraint_type] = code
                if constraint_type == "path":
                    for func in stage_dict[constraint_type]:
                        self.path_constraint_state[str(func)] = 0 # set inactivate
            self.constraint_fns[stage] = stage_dict
            self.constraint_fns_code[stage] = stage_dict_code
        return constraint_fns, constraint_fns_code
