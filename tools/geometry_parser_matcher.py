from perception.geometry_parser_matcher.geometry_parser_matcher import GeometryParserMatcher
import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="test.png", type=str)
parser.add_argument('--output', default="test_mask.png", type=str)
parser.add_argument("--description", type=str, default="the handle of the cup")
args = parser.parse_args()
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
mask = matcher.parse(args.input, args.description)
rgb = cv2.imread(args.input)
rgb[mask > 0] = [0, 0, 255]
cv2.imwrite(args.output, rgb)


