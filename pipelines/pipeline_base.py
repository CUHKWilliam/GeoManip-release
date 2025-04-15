import utils.builder as builder
from utils.registry import ROBOTS, CAMERAS, SOLVERS, VISUALIZERS, ENVIRONMENT

class PipelineBase():
    def __init__(self, config):
        pass
    
    def run(self,):
        raise NotImplementedError