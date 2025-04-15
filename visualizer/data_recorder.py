from utils.registry import VISUALIZERS

@VISUALIZERS.register_module()
class DataRecorderV1:
    def __init__(self, config):
        self.config = config
    
    def move_and_record_data(self, env):
        pass