import yaml

class ConfigDict(dict):
    def __getattr__(self, key):
        value = self.get(key)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path):
    with open(path, "r") as f:
        cfg_raw = yaml.safe_load(f)
    return ConfigDict(cfg_raw)
