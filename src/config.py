import yaml

def load_config(config_path):
    """加载YAML配置文件并返回字典。"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
