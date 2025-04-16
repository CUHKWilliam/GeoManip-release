import sys
import os
sys.path.append(os.getcwd())
from equipments.cameras.realsense.realsense import RealSense
from PIL import Image

config = {
    "resolution": [640, 480],
    "extrinsic": [[-0.18746593, 0.51792903,-0.83462929, 0.38557143], [ 0.98110535, 0.14011352,-0.13341847, 0.49040391], [0.04784155,-0.84387068,-0.53440945, 0.38047709], [0, 0, 0, 1]]
   
}
camera = RealSense(config)
color, depth = camera.update_image_depth()
Image.fromarray(color).save("obs.png")
