import base64
from openai import OpenAI
import os
import cv2
import json
import numpy as np
import time
from datetime import datetime
import parse
from utils.utils import encode_image
from utils.registry import GENERATORS

def load_functions_from_txt(txt_path, ):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    return  {"func": None, "code": functions_text}



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
            if "get_point_cloud(\"" in line:
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

@GENERATORS.register_module()
class GeometricAndCodeGenerator:
    def __init__(self, config):
        prompt_dir = config['prompt_dir']

        self.geometry_constraints_scheme_prompt = open(os.path.join(prompt_dir, 'geometry_constraints_scheme_prompt.txt'), "r").read()
        self.geometry_constraints_example_prompt = open(os.path.join(prompt_dir, 'geometry_constraints_example_prompt.txt'), "r").read()
        self.cost_functions_scheme_prompt = open(os.path.join(prompt_dir, 'cost_functions_scheme_prompt.txt'), "r").read()
        self.cost_functions_example_prompt = open(os.path.join(prompt_dir, 'cost_functions_example_prompt.txt'), "r").read()
        
        self.query_history = [{"role": "system", "content": "You are a helpful assistant that breaks down task, write geometric constraints, and write Python code for robot manipulation. Please learn from the knowledge as much as possible. Think carefully and reason step by step."}]

    def build_prompt_cost_functions(self,):
        prompt_text = self.cost_functions_scheme_prompt.format(self.cost_functions_example_prompt)

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

    def build_prompt_geometry_constraints(self, img, instruction, task_dir,):
        image_path = os.path.join(task_dir, 'query_img.png')
        cv2.imwrite(image_path, img)
        prompt_text = self.geometry_constraints_scheme_prompt.format(instruction, self.geometry_constraints_example_prompt)
        img_base64 = encode_image(image_path)
        message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    },
                    {
                        "type": "text",
                        "text": prompt_text
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
        import ipdb;ipdb.set_trace()
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

        
    def task_decomposition_and_geometric_constraint_generation(self, img, instruction, task_dir=None, overwrite=False):
        if task_dir is None:
            # create a directory for the task
            fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
            self.task_dir = os.path.join(self.base_dir, fname)
            os.makedirs(self.task_dir, exist_ok=True)
        prompt = self.build_prompt_geometry_constraints(img, instruction, task_dir)
        self.query_history.append(prompt)
        output_constraint_file = os.path.join(task_dir, "output_geometry_constraints.txt")
        self.task_dir = task_dir
        if not os.path.exists(output_constraint_file) or overwrite:
            # stream back the response
            output = self.constraint_generator_queryer.query(self.query_history, instruction, stream=True)
            with open(output_constraint_file, "w") as f:
                f.write(output)
        else:
            with open(output_constraint_file, "r") as f:
                output = f.read()
        self.query_history[1]['content'] = self.query_history[1]['content'][1:]
        self.query_history.append(
            {"role": "system", "content": "{}".format(output)}
        )
    
    def cost_fns_generation(self, instruction, task_dir=None, overwrite=False):
        prompt = self.build_prompt_cost_functions()
        self.query_history.append(prompt)
        output_cost_file = os.path.join(task_dir, "output_cost_functions.txt")
        if not os.path.exists(output_cost_file) or overwrite:
            output = self.cost_fns_queryer.query(self.query_history, instruction, stream=True)
            self.query_history.append(
                {"role": "system", "content": "{}".format(output)}
            )
            self.query_history.append(output)
            # save raw output
            with open(output_cost_file, 'w') as f:
                f.write(output)
        else:
            with open(output_cost_file, "r") as f:
                output = f.read()
        parse_and_save_cost_fns(output, self.task_dir)
        return output

class RekepKeypointConstraintGenerator:
    def __init__(self, ):
        config = {
            "model": "qwen2.5-vl-72b-instruct",
            "temperature": 0.,
            "max_tokens": 2048
        }
        self.config = config
        self.client = OpenAI(api_key="sk-b24ffb4725274f398d51d4aab97efe0d", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './vlm_query_rekep')
        with open(os.path.join(self.base_dir, 'prompt_template.txt'), 'r') as f:
            self.prompt_template = f.read()

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        prompt_text = self.prompt_template.format(instruction=instruction)
        # save prompt
        with open(os.path.join(self.task_dir, 'prompt.txt'), 'w') as f:
            f.write(prompt_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_template.format(instruction=instruction)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _parse_and_save_constraints(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start:end+1]
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
    
    def _parse_other_metadata(self, output):
        data_dict = dict()
        # find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        data_dict['num_stages'] = int(num_stages['num_stages'])
        # find grasp_keypoints
        grasp_keypoints_template = "grasp_keypoints = {grasp_keypoints}"
        for line in output.split("\n"):
            grasp_keypoints = parse.parse(grasp_keypoints_template, line)
            if grasp_keypoints is not None:
                break
        if grasp_keypoints is None:
            raise ValueError("grasp_keypoints not found in output")
        # convert into list of ints
        grasp_keypoints = grasp_keypoints['grasp_keypoints'].replace("[", "").replace("]", "").split(",")
        grasp_keypoints = [int(x.strip()) for x in grasp_keypoints]
        data_dict['grasp_keypoints'] = grasp_keypoints
        # find release_keypoints
        release_keypoints_template = "release_keypoints = {release_keypoints}"
        for line in output.split("\n"):
            release_keypoints = parse.parse(release_keypoints_template, line)
            if release_keypoints is not None:
                break
        if release_keypoints is None:
            raise ValueError("release_keypoints not found in output")
        # convert into list of ints
        release_keypoints = release_keypoints['release_keypoints'].replace("[", "").replace("]", "").split(",")
        release_keypoints = [int(x.strip()) for x in release_keypoints]
        data_dict['release_keypoints'] = release_keypoints
        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate(self, img, instruction, metadata):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # create a directory for the task
        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + instruction.lower().replace(" ", "_")
        self.task_dir = os.path.join(self.base_dir, fname)
        os.makedirs(self.task_dir, exist_ok=True)
        # save query image
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build prompt
        messages = self._build_prompt(image_path, instruction)
        # stream back the response
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                        messages=messages,
                                                        temperature=self.config['temperature'],
                                                        max_tokens=self.config['max_tokens'],
                                                        stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # parse and save constraints
        self._parse_and_save_constraints(output, self.task_dir)
        # save metadata
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        return self.task_dir
    

if __name__ == "__main__":
    prompt_path = "./vlm_query"
    generator = GeometricAndCodeGenerator(prompt_path)
    ## TODO: for naive test
    task_dir = "geomanip_test"
    os.makedirs(task_dir, exist_ok=True)
    rgb_path = "./geomanip_test.png"
    rgb = cv2.imread(rgb_path)
    print("Decomposing tasks and generate geometric constraint for each sub-task...")
    generator.task_decomposition_and_geometric_constraint_generation(rgb, "pour water from the cup to the bowl", task_dir=task_dir, overwrite=False)
    print("Generate cost function for each sub-task...")
    cost_function = generator.cost_fns_generation(task_dir=task_dir, overwrite=True)
    import ipdb;ipdb.set_trace()