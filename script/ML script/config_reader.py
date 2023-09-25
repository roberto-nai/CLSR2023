import yaml

yaml_file = "config.yml"

def read_yaml():
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)