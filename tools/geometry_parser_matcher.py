import sys
import os
sys.path.append(os.getcwd())
from perception.geometry_parser_matcher.geometry_parser_matcher import GeometryParserMatcher
import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="obs.png", type=str)
parser.add_argument('--output', default="obs_mask.png", type=str)
parser.add_argument("--description", type=str, default="plate")
args = parser.parse_args()
config = {
        'backbone': 'resnet50',
        'feature_extractor_path': 'saved_pretrained/resnet50_a1h-35c100f8.pth',
        'matcher_checkpoint_path': 'saved_pretrained/matcher.pt',
        "device": "cuda:0",
        "database_root": "database",
        "img_mean": [0.485, 0.456, 0.406],
        "img_std": [0.229, 0.224, 0.225],
        "img_size": 384,
        "max_retrieved_num": 30,
        "score_threshold": 0.6,
        "verbose": False
    }

matcher = GeometryParserMatcher(config)
mask = matcher.parse(args.input, args.description)
rgb = cv2.imread(args.input)
rgb[mask > 0] = [0, 0, 255]
cv2.imwrite(args.output, rgb)


