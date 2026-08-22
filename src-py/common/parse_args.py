import yaml

def parse_arguments(arguments):        
    args_dict = {}
    config_file_path = None
    for idx, item_ in enumerate(arguments):
        if item_ == '--run-config':
            config_file_path = arguments[idx + 1]
    assert config_file_path is not None, 'No config file provided! Aborting...'
    
    # Load YAML file with nested structure
    with open(config_file_path, 'r') as file:
        args_dict = yaml.safe_load(file)
    return args_dict

    