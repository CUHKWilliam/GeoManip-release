import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from large_models.queryer import WebQueryer
from generator.constraint_generator import GeometricAndCodeGenerator
import os
import cv2
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', type=str, default='queryer_database')
parser.add_argument("--prompt", type=str, default="./vlm_query")
parser.add_argument("--overwrite", type=bool, default=False)
args = parser.parse_args()
prompt = args.prompt
tasks = os.listdir(args.database_dir)

const_gen_config = {
    "prompt_dir": "vlm_query"
}
constraint_generator = GeometricAndCodeGenerator(
    config=const_gen_config,
)
cost_fns_querier_config = {
    "api_key": "sk-6cf616fbddf74b7ab8e79f8f764c4957",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "temperature": 0.8,
    "top_p": 0.5,
}
const_gen_querier_config = {
    "api_key": "sk-b24ffb4725274f398d51d4aab97efe0d",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen2.5-vl-72b-instruct",
    "temperature": 0.8,
    "top_p": 0.5,
}
cost_fns_querier = WebQueryer(
    config=cost_fns_querier_config,
)

constraint_generator_queirer = WebQueryer(
    config=const_gen_querier_config
)
constraint_generator.cost_fns_queryer = cost_fns_querier
constraint_generator.constraint_generator_queryer = constraint_generator_queirer

for task in tqdm(tasks):
    key_file = os.path.join(task, "key.txt")
    cost_fn_file = os.path.join(task, "output_cost_functions.txt")
    const_gen_file = os.path.join(task, "output_geometry_constraints.txt")
    if os.path.exists(key_file) and os.path.exists(cost_fn_file) and os.path.exists(const_gen_config) and args.overwrite:
        continue
    task_dir = os.path.join(args.database_dir, task)
    image_path = os.path.join(task_dir,  "query_image.png")
    if not os.path.exists():
        continue
    color_image = cv2.imread(os.path.join(task_dir, "query_image.png"))
    constraint_generator.task_decomposition_and_geometric_constraint_generation(color_image, task, task_dir=task_dir)
    cost_function_text = constraint_generator.cost_fns_generation(task, task_dir=task_dir)
    with open(key_file, "w") as f:
        f.write(task)