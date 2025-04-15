import inspect
import numpy as np


class Registry:
    _registry_instances = {}

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        Registry._registry_instances[name] = self

    def register_module(self, name=None, module=None):
        def _register(module):
            key = name or module.__name__
            if key in self._module_dict:
                raise ValueError(f'{key} is already registered in {self._name}')
            self._module_dict[key] = module
            print(f"register {module.__name__}")
            return module

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg):
        if isinstance(cfg, str):
            return self._module_dict[cfg]()
        elif isinstance(cfg, dict):
            args = cfg.copy()
            obj_type = args.pop('type')
            if isinstance(obj_type, str):
                obj_cls = self._module_dict[obj_type]
            elif inspect.isclass(obj_type):
                obj_cls = obj_type
            else:
                raise TypeError(
                    f'type must be a str or valid type, but got {type(obj_type)}')
            cls = obj_cls(args)
            for k in args.keys():
                if isinstance(args[k], list):
                    args[k] = np.array(args[k])
                setattr(cls, k, args[k])
            return cls
        else:
            raise TypeError(f'Invalid type {type(cfg)} for build')
    
    @classmethod
    def get_registry_map(cls):
        registry_map = {}
        for registry in cls._registry_instances.values():
            registry_map.update({k: registry for k in registry._module_dict.keys()})
        return registry_map
    
ROBOTS = Registry('robot')
CAMERAS = Registry('camera')
VISUALIZERS = Registry('visualizer')
SOLVERS = Registry('solver')
ENVIRONMENT = Registry('environment')
PERCEPTION = Registry('perception')
PIPELINES = Registry("pipeline")
GENERATORS = Registry("generator")