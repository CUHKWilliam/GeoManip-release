# geomanip/utils/builder.py

import yaml
from utils.registry import ROBOTS, CAMERAS, SOLVERS, VISUALIZERS, Registry

def build_component(registry, config):
    registry_map = Registry.get_registry_map()
    if not isinstance(config, dict):
        return config

    if 'type' in config:
        component_type = config['type']
        component_config = config.get('config', {})
        
        # Determine which registry to use based on the component type
        registry = registry_map.get(component_type, None)
        if registry is None:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Recursively build sub-components
        for key, value in component_config.items():
            component_config[key] = build_component(registry_map, value)
        return registry.build(dict(type=component_type, **component_config))
    else:
        # If no 'type' key, recursively process each item in the dict
        return {k: build_component(registry_map, v) for k, v in config.items()}

