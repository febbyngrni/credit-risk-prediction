import yaml
import joblib

def dump_joblib(data, path: str) -> None:
    joblib.dump(data, path)

def load_joblib(path: str):
    return joblib.load(path)

def load_params(params_dir):
    with open(params_dir, 'r') as file:
        params = yaml.safe_load(file)

    return params