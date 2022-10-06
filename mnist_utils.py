import yaml
import argparse


def moving_average(array, window_size):
    i = 0
    moving_averages = []
    while i < len(array) - window_size + 1:
        this_window = array[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def load_yaml(file_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='param_mnist')
    args = parser.parse_args()
    if file_name:
        yaml_file = f'config/{file_name}.yaml'
    else:
        yaml_file = f'config/{args.yaml}.yaml'
    with open(yaml_file) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)
    return param
