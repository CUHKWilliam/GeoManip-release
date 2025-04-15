from utils.registry import PIPELINES
from .base_grasper import GrasperBase

@PIPELINES.register_module()
class PoseEstimationGrasper(GrasperBase):
    def __init__(self, config):
        self.config = config
    
    def grasp(self, env):
        pass