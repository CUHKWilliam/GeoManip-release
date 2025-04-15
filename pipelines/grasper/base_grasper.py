import cv2
import numpy as np  
class GrasperBase:
    def __init__(self, config):
        self.config = config
    
    def grasp(self, env):
        return NotImplementedError