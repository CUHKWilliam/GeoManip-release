

import numpy as np
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from utils import *
import cv2
import argparse
import yaml
import utils.builder as builder
from utils.registry import PIPELINES
from visualizer import *
from pipelines import *
from generator import *
from perception import *
from envs import *
from equipments import *
from large_models import *


parser = argparse.ArgumentParser()
# parser.add_argument("--task", type=str, default="Pour water from the cup to the bowl")
# parser.add_argument("--task_dir", type=str, default="vlm_query/pour_water_from_cup_to_bowl")
# parser.add_argument("--config_path", type=str, default="configs/geomanip_config_realworld_improved.yaml")

parser.add_argument("--task", type=str, default="Put the kiwi onto the plate")
parser.add_argument("--task_dir", type=str, default="vlm_query/place_kiwi_onto_plate")
parser.add_argument("--config_path", type=str, default="configs/geomanip_config_realworld_improved.yaml")


parser.add_argument("--device", type=str, default="mps")
args = parser.parse_args()


def merge_configs(yaml_config, cli_args):
    cli_config = vars(cli_args)
    def recursive_merge(yaml_dict, key, value):
        yaml_dict[key] = value
        for key2 in yaml_dict.keys():
            if isinstance(yaml_dict[key2], dict):
                recursive_merge(yaml_dict[key2], key, value)
            
    for key, value in cli_config.items():
        if value is not None and key != 'config':
            recursive_merge(yaml_config, key, value)
    
    return yaml_config

if __name__ == "__main__":
    task = args.task
    task_dir = args.task_dir
    config_path = args.config_path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = merge_configs(config, args)
    inference_pipeline = builder.build_component(PIPELINES, config['pipeline'])
    inference_pipeline.run()