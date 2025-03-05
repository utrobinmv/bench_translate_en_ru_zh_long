import yaml

def load_config(config_filename)
    with open(config_filename, 'r') as file:
        return yaml.safe_load(file)
    return None
